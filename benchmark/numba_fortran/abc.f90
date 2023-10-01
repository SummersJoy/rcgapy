module abc
use iso_c_binding, only: c_int, c_double
implicit none
integer, parameter :: dp = kind(0d0)
private
public c_abc_model_fortran

contains


subroutine c_abc_model_fortran(n, a, b, c, inflow, outflow) bind(c)
integer(c_int), intent(in), value :: n
real(c_double), intent(in), value :: a, b, c
real(c_double), intent(in) :: inflow(n)
real(c_double), intent(out) :: outflow(n)
call abc_model(a, b, c, inflow, outflow)
end subroutine


subroutine abc_model(a, b, c, inflow, outflow)
real(dp), intent(in) :: a, b, c, inflow(:)
real(dp), intent(out) :: outflow(:)
real(dp) :: state_in, state_out
integer :: t
state_in = 0
do t = 1, size(inflow)
    state_out = (1 - c) * state_in + a * inflow(t)
    outflow(t) = (1 - a - b) * inflow(t) + c * state_in
    state_in = state_out
end do
end subroutine


end module