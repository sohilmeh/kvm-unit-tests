/*
 * Intel LASS (Linear Address Space Separation) unit test
 *
 * Copyright (C) 2026 Intel
 *
 * This work is licensed under the terms of the GNU LGPL, version 2 or
 * later.
 *
 * LASS splits the linear address space in half using bit 63 and, when
 * CR4.LASS=1, faults accesses that cross the split without consulting the
 * paging structures: user-mode accesses to the supervisor half, supervisor
 * instruction fetches from the user half, and supervisor data accesses to the
 * user half that CR4.SMAP would reject as well.  Violations are #GP(0), or
 * #SS(0) for stack accesses, and take priority over the paging permission
 * checks, i.e. a violation reports #GP where SMAP or a U/S mismatch alone
 * reports #PF.  Every case below runs with LASS off and then on, so that
 * enforcement is distinguished from the paging checks.
 *
 * Note, kvm-unit-tests links its image low and identity maps it, i.e. runs
 * CPL0 code, stacks and descriptor tables in the user half, where setting
 * CR4.LASS faults the next instruction fetch and triple faults the guest.
 * The test therefore aliases the image into the supervisor half and runs the
 * LASS-enabled body from there, the way a 64-bit kernel is laid out.
 */

#include "libcflat.h"
#include "processor.h"
#include "desc.h"
#include "msr.h"
#include "setjmp.h"
#include <util.h>
#include "vmalloc.h"
#include "alloc_page.h"
#include "vm.h"
#include "asm/io.h"
#include "ioram.h"

/*
 * Offset of the image's supervisor alias.  Bits 63:44 are all ones so that
 * aliased addresses are canonical under both 4- and 5-level paging, and so
 * that the range stays clear of the vpages alloc_vpage() hands out.
 */
#define SUP_OFFSET	0xffffff8000000000UL

static unsigned long to_sup(const void *addr)
{
	unsigned long a = (unsigned long)addr;

	return (a & BIT_ULL(63)) ? a : a + SUP_OFFSET;
}

static unsigned long to_usr(const void *addr)
{
	unsigned long a = (unsigned long)addr;

	return (a & BIT_ULL(63)) ? a - SUP_OFFSET : a;
}

/*
 * Exception capture that works from either alias.  ASM_TRY() records absolute
 * RIPs in .data.ex, which never match code running at the supervisor alias,
 * so resume instead at a fixup address the probe computes RIP-relatively.
 */
static void *fault_fixup;
static unsigned int fault_vector;
static unsigned long fault_rip;

/*
 * LASS enforcement of supervisor data accesses is gated on CR4.SMAP, and
 * CLAC/STAC #UD without it, so cases that need SMAP=1 are skipped if the CPU
 * doesn't support SMAP.
 */
static bool has_smap;

/*
 * The CR4 bits to add while a probe runs.  Like SMAP, LASS can only be
 * enabled across the probe itself, never across a report(): vsnprintf()
 * dispatches a switch through a jump table of absolute addresses, so calling
 * it from the supervisor alias jumps back into the user half mid-function,
 * which is a supervisor fetch from a user address.
 */
static unsigned long lass_window(bool lass_on)
{
	return lass_on ? X86_CR4_LASS : 0;
}

static void lass_fault_handler(struct ex_regs *regs)
{
	/*
	 * Only claim faults a probe is expecting.  Anything else is a bug in
	 * the test rather than in LASS, and resuming at a stale fixup turns it
	 * into an unrecoverable triple fault instead of a readable dump.
	 */
	if (!fault_fixup) {
		/*
		 * Close the window before dumping state, as unhandled_exception()
		 * reads user-half format strings, i.e. would fault again and
		 * hang the vCPU instead of printing anything useful.
		 */
		write_cr4(read_cr4() & ~(X86_CR4_LASS | X86_CR4_SMAP));
		unhandled_exception(regs, false);
		return;
	}

	fault_vector = regs->vector;
	fault_rip = regs->rip;
	regs->rip = (unsigned long)fault_fixup;
	fault_fixup = NULL;
}

#define __lass_safe(prefix, insn, inputs...)				\
({									\
	fault_vector = 0;						\
	asm volatile("lea 88f(%%rip), %%r11\n\t"			\
		     "mov %%r11, %[fixup]\n\t"				\
		     prefix insn "\n\t"					\
		     "88:\n\t"						\
		     : [fixup] "=m" (fault_fixup)			\
		     : inputs						\
		     : "r10", "r11", "memory");				\
	fault_fixup = NULL;						\
	fault_vector;							\
})

#define lass_safe(insn, inputs...)	__lass_safe("", insn, inputs)
#define lass_fep_safe(insn, inputs...)	__lass_safe(KVM_FEP, insn, inputs)

#define lass_maybe_fep_safe(fep, insn, inputs...)			\
	((fep) ? lass_fep_safe(insn, inputs) : lass_safe(insn, inputs))

static unsigned int lass_read(volatile u64 *ptr, bool fep)
{
	return lass_maybe_fep_safe(fep, "mov (%[ptr]), %%r10", [ptr] "r" (ptr));
}

static unsigned int lass_write(volatile u64 *ptr, bool fep)
{
	return lass_maybe_fep_safe(fep, "mov %[val], (%[ptr])",
				   [ptr] "r" (ptr), [val] "r" (0ul));
}

static unsigned int lass_invlpg(volatile u64 *ptr, bool fep)
{
	return lass_maybe_fep_safe(fep, "invlpg (%[ptr])", [ptr] "r" (ptr));
}

/*
 * Test pages.  Supervisor-half mappings get U/S=0 so that SMAP can't fire on
 * them, leaving LASS as the only thing that can fault a CPL0 access.  The
 * user-half pages keep the identity map's U/S=1, so a CPL0 access with SMAP=1
 * and AC=0 faults as #GP if LASS is enforced, and as #PF if it isn't.
 */
static volatile u64 *sup_ram;
static volatile u64 *usr_ram;
static volatile u64 *sup_mmio;
static volatile u64 *usr_mmio;

static void *alloc_sup_page(phys_addr_t phys)
{
	void *va = alloc_vpage();

	install_page(current_page_table(), phys, va);
	*get_pte(current_page_table(), va) &= ~PT_USER_MASK;
	invlpg(va);

	return va;
}

static void setup_test_pages(void)
{
	sup_ram = alloc_sup_page(virt_to_phys(alloc_page()));
	sup_mmio = alloc_sup_page(IORAM_BASE_PHYS);
	usr_ram = alloc_page();
	usr_mmio = phys_to_virt(IORAM_BASE_PHYS);
}

/* Alias the whole image into the supervisor half, with U/S=0. */
static void map_supervisor_alias(void)
{
	extern char edata;
	unsigned long end, phys;

	end = ((unsigned long)&edata + LARGE_PAGE_SIZE - 1) &
	      ~(unsigned long)(LARGE_PAGE_SIZE - 1);

	for (phys = 0; phys < end; phys += LARGE_PAGE_SIZE) {
		pteval_t *pte = install_large_page(current_page_table(), phys,
						   (void *)(phys + SUP_OFFSET));
		*pte &= ~PT_USER_MASK;
	}
}

static struct descriptor_table_ptr usr_idtr, usr_gdtr;
static unsigned long usr_idt_offset[32];
static unsigned long usr_gs_base, usr_tss_base;
static u16 tr_sel;

static void idt_set_offset(int vec, unsigned long addr)
{
	boot_idt[vec].offset0 = addr;
	boot_idt[vec].offset1 = addr >> 16;
	boot_idt[vec].offset2 = addr >> 32;
}

/*
 * Repoint everything the CPU touches implicitly at the supervisor alias:
 * exception delivery reads the IDT and GDT and then fetches the handler, the
 * CPL3 cases need the TSS, and per-CPU state lives at GS_BASE.  Note, RSP0 is
 * published later, by run_at_cpl3().
 */
static void enter_supervisor_world(void)
{
	struct descriptor_table_ptr dt;
	int i;

	for (i = 0; i < 32; i++) {
		if (!boot_idt[i].p)
			continue;
		usr_idt_offset[i] = (unsigned long)get_idt_addr(&boot_idt[i]);
		idt_set_offset(i, to_sup((void *)usr_idt_offset[i]));
	}

	sidt(&usr_idtr);
	dt = usr_idtr;
	dt.base = to_sup((void *)dt.base);
	lidt(&dt);

	sgdt(&usr_gdtr);
	dt = usr_gdtr;
	dt.base = to_sup((void *)dt.base);
	lgdt(&dt);

	usr_gs_base = rdmsr(MSR_GS_BASE);
	wrmsr(MSR_GS_BASE, to_sup((void *)usr_gs_base));

	tr_sel = str();
	usr_tss_base = get_gdt_entry_base(get_tss_descr());
	set_gdt_entry_base(tr_sel, to_sup((void *)usr_tss_base));
	clear_tss_busy(tr_sel);
	ltr(tr_sel);
}

static void leave_supervisor_world(void)
{
	int i;

	set_gdt_entry_base(tr_sel, usr_tss_base);
	clear_tss_busy(tr_sel);
	ltr(tr_sel);
	tss[0].rsp0 = 0;

	wrmsr(MSR_GS_BASE, usr_gs_base);
	lgdt(&usr_gdtr);
	lidt(&usr_idtr);

	for (i = 0; i < 32; i++) {
		if (boot_idt[i].p)
			idt_set_offset(i, usr_idt_offset[i]);
	}
}

/*
 * Call @fn at its supervisor alias, with the stack switched to the alias of
 * the current stack.  @fn owns CR4.LASS and must leave it clear, as the
 * return address here is a user-half address.
 */
static void run_in_supervisor(void (*fn)(void))
{
	unsigned long sup_fn = to_sup((void *)fn);

	asm volatile("mov %%rsp, %%rbx\n\t"
		     "add %[off], %%rsp\n\t"
		     "call *%[fn]\n\t"
		     "mov %%rbx, %%rsp\n\t"
		     :
		     : [fn] "r" (sup_fn), [off] "r" (SUP_OFFSET)
		     : "rbx", "memory");
}

/*
 * Landing pad for the supervisor-fetch case.  LASS reports the violation on
 * the branch target rather than the branch, so if LASS is off, or broken, the
 * jump lands here and needs a way back.
 */
static unsigned long fetch_recover;

extern char fetch_probe[];
asm(".pushsection .text\n\t"
    "fetch_probe:\n\t"
    "jmp *fetch_recover(%rip)\n\t"
    ".popsection");

static void test_sup_fetch(bool lass_on)
{
	unsigned long target = to_usr(fetch_probe);
	unsigned long cr4 = read_cr4();

	fault_vector = 0;
	fault_rip = 0;

	write_cr4(cr4 | lass_window(lass_on));
	asm volatile("lea 88f(%%rip), %%r11\n\t"
		     "mov %%r11, %[fixup]\n\t"
		     "mov %%r11, %[recover]\n\t"
		     "jmp *%[target]\n\t"
		     "88:\n\t"
		     : [fixup] "=m" (fault_fixup), [recover] "=m" (fetch_recover)
		     : [target] "r" (target)
		     : "r11", "memory");
	fault_fixup = NULL;
	write_cr4(cr4);

	report(fault_vector == (lass_on ? GP_VECTOR : 0),
	       "CPL0 fetch from user address: expected %s, got vector %u",
	       lass_on ? "#GP" : "no fault", fault_vector);

	if (lass_on && fault_vector == GP_VECTOR)
		report(fault_rip == target,
		       "LASS violation reported on the branch target");
}

/*
 * CPL0 data accesses.  Note, all reporting happens outside the window the
 * probes run in: with SMAP=1 and AC=0 the format strings and varargs are
 * user-half pointers on U/S=1 pages, and with LASS on the library code itself
 * is unreachable (see lass_window()).
 */
static void test_sup_data(const char *what, volatile u64 *sup,
			  volatile u64 *usr, bool lass_on, bool fep)
{
	unsigned long cr4 = read_cr4();
	unsigned int v_sup, v_smap0, v_ac0, v_ac1;

	write_cr4(cr4 | lass_window(lass_on));
	v_sup = lass_read(sup, fep);
	v_smap0 = lass_read(usr, fep);
	write_cr4(cr4);

	report(!v_sup, "%s: CPL0 access to supervisor address: no fault (got %u)",
	       what, v_sup);
	report(!v_smap0, "%s: CPL0 access to user address, SMAP=0: no fault (got %u)",
	       what, v_smap0);

	if (!has_smap) {
		report_skip("%s: CPL0 access to user address with SMAP=1", what);
		return;
	}

	write_cr4(cr4 | lass_window(lass_on) | X86_CR4_SMAP);
	clac();
	v_ac0 = lass_read(usr, fep);
	stac();
	v_ac1 = lass_read(usr, fep);
	clac();
	write_cr4(cr4);

	report(v_ac0 == (lass_on ? GP_VECTOR : PF_VECTOR),
	       "%s: CPL0 access to user address, SMAP=1 AC=0: expected %s, got %u",
	       what, lass_on ? "#GP (LASS)" : "#PF (SMAP)", v_ac0);
	report(!v_ac1, "%s: CPL0 access to user address, SMAP=1 AC=1: no fault (got %u)",
	       what, v_ac1);
}

/*
 * TLB-invalidation operands are exempt from LASS in either half.  Run with
 * SMAP=1 and AC=0, i.e. under exactly the conditions that make a plain data
 * access to the user-half address fault, so that the exemption is what is
 * actually under test.
 */
static void test_invlpg(bool lass_on, bool fep)
{
	unsigned long cr4 = read_cr4();
	unsigned int v_sup, v_usr;

	if (!has_smap) {
		report_skip("%sINVLPG LASS exemption", fep ? "fep: " : "");
		return;
	}

	write_cr4(cr4 | lass_window(lass_on) | X86_CR4_SMAP);
	clac();
	v_sup = lass_invlpg(sup_ram, fep);
	v_usr = lass_invlpg(usr_ram, fep);
	write_cr4(cr4);

	report(!v_sup && !v_usr,
	       "%sINVLPG on supervisor and user addresses: no fault (got %u, %u)",
	       fep ? "fep: " : "", v_sup, v_usr);
}

/*
 * A supervisor data write must fault too, not just a read, and the emulator
 * has a separate path for writes.
 */
static void test_sup_write(bool lass_on, bool fep)
{
	unsigned long cr4 = read_cr4();
	unsigned int vector;

	if (!has_smap) {
		report_skip("%sCPL0 write to user address with SMAP=1",
			    fep ? "fep: " : "");
		return;
	}

	write_cr4(cr4 | lass_window(lass_on) | X86_CR4_SMAP);
	clac();
	vector = lass_write(usr_ram, fep);
	write_cr4(cr4);

	report(vector == (lass_on ? GP_VECTOR : PF_VECTOR),
	       "%sCPL0 write to user address, SMAP=1 AC=0: expected %s, got %u",
	       fep ? "fep: " : "", lass_on ? "#GP (LASS)" : "#PF (SMAP)", vector);
}

/*
 * CPL3 probes.  User code has to execute from the user alias, so the probe is
 * entered at its low address; it records the vector itself and leaves CPL3 via
 * #UD, which the handler below turns back into a normal return.
 */
static jmp_buf user_jmpbuf;
static volatile u64 *user_target;
static bool user_fep;
static char user_stack[4096] __attribute__((aligned(16)));

static noinline void user_probe(void)
{
	if (user_fep)
		lass_fep_safe("mov (%[ptr]), %%r10", [ptr] "r" (user_target));
	else
		lass_safe("mov (%[ptr]), %%r10", [ptr] "r" (user_target));

	asm volatile("ud2");
}

static void user_return_trampoline(void)
{
	longjmp((struct jmp_buf_tag *)to_sup(user_jmpbuf), 1);
}

/*
 * Return to CPL0 the way usermode.c does: redirect to a trampoline rather than
 * longjmp()ing from the handler, so the longjmp happens after the IRET.  SS has
 * to be reloaded explicitly because the CPU nulls it when handling an exception
 * changes the CPL.
 */
static void user_exit_handler(struct ex_regs *regs)
{
	regs->rip = to_sup((void *)user_return_trampoline);
	regs->cs = KERNEL_CS;
	regs->ss = KERNEL_DS;
	regs->rsp = tss[0].rsp0;
	regs->rflags &= ~X86_EFLAGS_AC;
}

static unsigned int run_at_cpl3(volatile u64 *target, bool fep)
{
	user_target = target;
	user_fep = fep;
	fault_vector = 0;

	if (setjmp(user_jmpbuf))
		return fault_vector;

	/*
	 * setup_tss() zeroes the TSS on x86_64, so publish a supervisor-half
	 * RSP0 for the CPL3 -> CPL0 stack switch.  The current RSP already
	 * points at the supervisor alias of the stack.
	 */
	asm volatile("mov %%rsp, %[rsp0]\n\t"
		     "pushq %[uds]\n\t"
		     "pushq %[ustack]\n\t"
		     "pushfq\n\t"
		     "pushq %[ucs]\n\t"
		     "pushq %[urip]\n\t"
		     "iretq\n\t"
		     : [rsp0] "=m" (tss[0].rsp0)
		     : [uds] "i" (USER_DS), [ucs] "i" (USER_CS),
		       [ustack] "r" (to_usr(user_stack) + sizeof(user_stack)),
		       [urip] "r" (to_usr((void *)user_probe))
		     : "memory");

	__builtin_unreachable();
}

static void test_lass_user(bool lass_on, bool fep)
{
	const char *pfx = fep ? "fep: " : "";
	unsigned long cr4 = read_cr4();
	unsigned int v_usr, v_sup;

	write_cr4(cr4 | lass_window(lass_on));
	v_usr = run_at_cpl3(usr_ram, fep);
	v_sup = run_at_cpl3(sup_ram, fep);
	write_cr4(cr4);

	report(!v_usr, "%sCPL3 access to user address: no fault (got %u)",
	       pfx, v_usr);
	report(v_sup == (lass_on ? GP_VECTOR : PF_VECTOR),
	       "%sCPL3 access to supervisor address: expected %s, got %u",
	       pfx, lass_on ? "#GP (LASS)" : "#PF (U/S)", v_sup);
}

/*
 * Runs from the supervisor alias, where enabling LASS is survivable.  Uses
 * lass_safe() rather than write_cr4_safe() because ASM_TRY() can't catch
 * faults raised at the alias.
 */
static void test_cr4_lass_toggle(void)
{
	unsigned long cr4 = read_cr4();
	unsigned int v_set, v_clear;
	bool set, clear;

	v_set = lass_safe("mov %[val], %%cr4", [val] "r" (cr4 | X86_CR4_LASS));
	set = !!(read_cr4() & X86_CR4_LASS);

	v_clear = lass_safe("mov %[val], %%cr4", [val] "r" (cr4));
	clear = !(read_cr4() & X86_CR4_LASS);

	/*
	 * Both writes are the thing under test, so neither can report while
	 * the other is still pending.  Force LASS off if the clear did not
	 * take, or the reports below would fault instead of printing.
	 */
	if (!clear)
		write_cr4(cr4);

	report(!v_set && set,
	       "Expected CR4.LASS=1 to succeed, got vector %u", v_set);
	report(!v_clear && clear,
	       "Expected CR4.LASS=0 to succeed, got vector %u", v_clear);
}

static void run_matrix(bool lass_on)
{
	test_sup_data("native", sup_ram, usr_ram, lass_on, false);

	/*
	 * MMIO drives KVM's emulator for the cases that do not violate.  The
	 * violating case here is resolved by hardware before KVM sees it, so it
	 * asserts the architectural result rather than KVM's check.
	 */
	test_sup_data("mmio", sup_mmio, usr_mmio, lass_on, false);
	test_sup_write(lass_on, false);
	test_invlpg(lass_on, false);
	test_sup_fetch(lass_on);
	test_lass_user(lass_on, false);

	if (!is_fep_available) {
		report_skip("Forced emulation unavailable; emulator paths not tested");
		return;
	}

	test_sup_data("fep", sup_ram, usr_ram, lass_on, true);
	test_sup_write(lass_on, true);
	test_invlpg(lass_on, true);
	test_lass_user(lass_on, true);
}

/* Runs from the supervisor alias, on the alias of the boot stack. */
static void supervisor_body(void)
{
	unsigned long rip;

	asm volatile("lea 0(%%rip), %[rip]" : [rip] "=r" (rip));
	report(!!(rip & BIT_ULL(63)),
	       "Executing supervisor code from the upper half (RIP 0x%lx)", rip);

	/*
	 * Run the matrix with LASS off first.  It is a control group - every
	 * case below must fault differently, or not at all, once LASS is on -
	 * and it validates the high-half plumbing on CPUs without LASS.
	 */
	report_prefix_push("lass=0");
	run_matrix(false);
	report_prefix_pop();

	if (!this_cpu_has(X86_FEATURE_LASS)) {
		report_skip("LASS unsupported; enforcement not tested");
		return;
	}

	test_cr4_lass_toggle();

	/*
	 * Each case enables LASS around its own probes rather than holding it
	 * across the matrix, because reporting a result needs it off again.
	 */
	report_prefix_push("lass=1");
	run_matrix(true);
	report_prefix_pop();
}

/*
 * CR4.LASS can only be set if LASS is enumerated.  Note, this runs in the user
 * world, where it is only safe *because* the write is expected to fail: if the
 * CPU accepted CR4.LASS here, the next instruction fetch would violate LASS
 * and triple fault the guest.  The accept case is therefore tested from the
 * supervisor alias instead, in test_cr4_lass_toggle().
 */
static void test_cr4_lass_unsupported(void)
{
	unsigned long cr4 = read_cr4();
	int vector;

	if (this_cpu_has(X86_FEATURE_LASS))
		return;

	vector = write_cr4_safe(cr4 | X86_CR4_LASS);
	report(vector == GP_VECTOR,
	       "Expected CR4.LASS=1 to #GP without CPUID.LASS, got vector %d",
	       vector);
	report(!(read_cr4() & X86_CR4_LASS), "CR4.LASS left clear");
}

int main(int ac, char **av)
{
	handler old_gp, old_pf, old_ss, old_ud;

	setup_vm();

	has_smap = this_cpu_has(X86_FEATURE_SMAP);
	if (!has_smap)
		report_info("SMAP unsupported; LASS enforcement of supervisor-mode data accesses is not tested");

	test_cr4_lass_unsupported();

	setup_test_pages();
	map_supervisor_alias();

	old_gp = handle_exception(GP_VECTOR, (handler)to_sup((void *)lass_fault_handler));
	old_pf = handle_exception(PF_VECTOR, (handler)to_sup((void *)lass_fault_handler));
	old_ss = handle_exception(SS_VECTOR, (handler)to_sup((void *)lass_fault_handler));
	old_ud = handle_exception(UD_VECTOR, (handler)to_sup((void *)user_exit_handler));

	enter_supervisor_world();
	run_in_supervisor(supervisor_body);
	leave_supervisor_world();

	handle_exception(GP_VECTOR, old_gp);
	handle_exception(PF_VECTOR, old_pf);
	handle_exception(SS_VECTOR, old_ss);
	handle_exception(UD_VECTOR, old_ud);

	return report_summary();
}
