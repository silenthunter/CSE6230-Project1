import math

l1_size = 32 * 1024

float_size = 8

m = n = k = 512     # per-matrix size: 2 M; total size: 6.2 M
#m = n = k = 256     # per-matrix size: 520 K; total size: 1.5 M

a_size = m * n * float_size
b_size = m * n * float_size
c_size = m * n * float_size

total_size = a_size + b_size + c_size

#print 'a_size=', a_size
#print 'total_size=', total_size

inner_loop_memory_sizes = []

for bk_log in range(0, int(math.log(k, 2) + 1)):
  for bm_log in range(0, int(math.log(m, 2) + 1)):
    for bn_log in range(0, int(math.log(n, 2) + 1)):

        # Calcuate block sizes in bytes
        bk = 2 ** bk_log
        bm = 2 ** bm_log
        bn = 2 ** bn_log

        # Number of blocks
        k_blocks = k / bk
        m_blocks = m / bm
        n_blocks = n / bn

        a_blocked_size = bm * bk * float_size
        b_blocked_size = bk * bn * float_size
        c_blocked_size = bm * bn * float_size

        inner_loop_memory_size = a_blocked_size + b_blocked_size + c_blocked_size
        inner_loop_memory_sizes.append(
            [inner_loop_memory_size, (
                bk, bm, bn)])

inner_loop_memory_sizes.sort(key=lambda x: x[0])

for x in inner_loop_memory_sizes:
  print x

# Block widths (6160 bytes for 256x256x256)
#bk = 1
#bm = 2
#bn = 256

#
## Number of blocks
#k_blocks = k / bk
#m_blocks = m / bm
#n_blocks = n / bn
#
#a_blocked_size = bm * bk * float_size
#b_blocked_size = bk * bn * float_size
#c_blocked_size = bm * bn * float_size
#
#print 'Block sizes'
#print 'a_blocked_size', a_blocked_size
#print 'b_blocked_size', b_blocked_size
#print 'c_blocked_size', c_blocked_size

#print 'Inner loop memory size=', a_blocked_size + b_blocked_size + c_blocked_size
