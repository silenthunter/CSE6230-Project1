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

$unrolled .= "if(K % BLOCK_SIZE != 0) return;\n";
$unrolled .= "int i, j, k;\nint bs = BLOCK_SIZE;\n";
$unrolled .= "__m128 a, b, a1, b1, prod0, tempDot0, prod1, tempDot1;\n";
$unrolled .= "__declspec(align(16)) float prodResult[4];\n";
$unrolled .= "float dotProd;\n\n";

$unrolled .= "for(i = 0; i < $N; i++)\n{\n";
#for(my $i = 0; $i < 48; $i++)
#{
	$unrolled .= "for(j = 0; j < $N; j++)\n{\n";
	#for(my $j = 0; $j < $N; $j++)
	#{
		$unrolled .= "dotProd = 0.0;\nprod0 = _mm_set1_ps(0);\nprod1 = _mm_set1_ps(0);\n\n";
		
		my $alt = 1;
		for(my $k = 0; $k < $N; $k+=4)
		{
			my $preA = "a";
			my $preB = "b";
			if($alt > 0)
			{
				$preA .= $alt;
				$preB .= $alt;
			}
			$unrolled .= "$preA = _mm_load_ps(&A[i * K + " .$k."]);\n";
			$unrolled .= "$preB = _mm_load_ps(&B[j * K + ".$k."]);\n";
			
			$unrolled .= "tempDot$alt = _mm_mul_ps($preA, $preB);\n";
			$unrolled .= "prod$alt = _mm_add_ps(prod$alt, tempDot$alt);\n";
			$alt += 1;
			if($alt > 1)
			{
				$alt = 0;
			}
		}
		
		$unrolled .= "prod0 = _mm_add_ps(prod0, prod1);\n";
		$unrolled .= "_mm_store_ps(prodResult, prod0);\n";
		$unrolled .= "dotProd += prodResult[0] + prodResult[1] + prodResult[2] + prodResult[3];\n";
		#my $cIndex = $j * $N;
		$unrolled .= "C[j * lda + i] = (double)dotProd + C[j * lda + i];\n\n";
	#}
	$unrolled .= "}\n";
#}
$unrolled .= "}\n";

$buffer =~ s/\{replace\}/$unrolled/;
$buffer =~ s/#define BLOCK_SIZE \d\d/#define BLOCK_SIZE $N/;

print FILEOUT $buffer;
