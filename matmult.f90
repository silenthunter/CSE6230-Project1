subroutine matmult (n,A,B,C)
  implicit none
  integer, intent(in) :: n
  double precision, intent(in) :: A(n*n)
  double precision, intent(in) :: B(n*n)
  double precision, intent(out) :: C(n*n)


  integer ierr, row, col, k_iter, a_index, b_index, c_index
  double precision dotprod

  do col = 0, n-1
    do row = 0, n-1
      dotprod = 0
      do k_iter =  0, n-1
        a_index = (k_iter * n) + row
        b_index = (col * n) + k_iter
        dotprod = dotprod + A(a_index + 1) * B(b_index + 1)
      end do
      c_index = (col * n) + row
      C(c_index + 1) = dotprod + C(c_index+1)
    end do
  end do


end subroutine
