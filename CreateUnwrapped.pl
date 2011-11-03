use strict;
my $N = 64;

open(FILEIN, "matmult_blocked_template.c");

my $buffer = "";
while(<FILEIN>)
{
	$buffer .= $_;
}
print "Unrolling loop\n";

close(FILEIN);

open(FILEOUT, ">matmult_blocked_copy.c");

my $unrolled = "";

$unrolled .= "int i, j, k;\nint bs = BLOCK_SIZE;\n";
$unrolled .= "__m128d a, b, a1, b1, b2, b12, prod, tempDot, prod1, tempDot1;\n";
$unrolled .= "__declspec(align(16)) double prodResult[2];\n";
$unrolled .= "double dotProd;\n\n";

$unrolled .= "for(i = 0; i < $N; i++)\n{\n";
#for(my $i = 0; $i < 48; $i++)
#{
	$unrolled .= "for(j = 0; j < $N; j+=2)\n{\n";
	#for(my $j = 0; $j < $N; $j++)
	#{
		$unrolled .= "dotProd = 0.0;\nprod = _mm_set1_pd(0);\nprod1 = _mm_set1_pd(0);\n\n";
		
		my $alt = 1;
		for(my $k = 0; $k < $N; $k+=2)
		{
			my $preA = "a";
			my $preB = "b";
			if($alt > 0)
			{
				$preA .= $alt;
				$preB .= $alt;
			}
			$unrolled .= "$preA = _mm_load_pd(&A[i * K + " .$k."]);\n";
			$unrolled .= "$preB = _mm_load_pd(&B[j * K + ".$k."]);\n";
			$unrolled .= $preB."2 = _mm_load_pd(&B[(j + 1) * K + ".$k."]);\n";
			
			$unrolled .= "tempDot = _mm_mul_pd($preA, $preB);\n";
			$unrolled .= "prod = _mm_add_pd(prod, tempDot);\n";
			$unrolled .= "tempDot1 = _mm_mul_pd($preA, ".$preB."2);\n";
			$unrolled .= "prod1 = _mm_add_pd(prod, tempDot1);\n";
			$alt += 1;
			if($alt > 1)
			{
				$alt = 0;
			}
		}
		
		$unrolled .= "_mm_store_pd(prodResult, prod);\n";
		$unrolled .= "dotProd += prodResult[0] + prodResult[1];\n";
		$unrolled .= "C[j * lda + i] = dotProd + C[j * lda + i];\n\n";
		$unrolled .= "_mm_store_pd(prodResult, prod1);\n";
		$unrolled .= "dotProd = 0;\n";
		$unrolled .= "dotProd += prodResult[0] + prodResult[1];\n";
		#my $cIndex = $j * $N;
		$unrolled .= "C[(j + 1) * lda + i] = dotProd + C[(j + 1) * lda + i];\n\n";
	#}
	$unrolled .= "}\n";
#}
$unrolled .= "}\n";

$buffer =~ s/\{replace\}/$unrolled/;
$buffer =~ s/#define BLOCK_SIZE \d\d/#define BLOCK_SIZE $N/;

print FILEOUT $buffer;
