A = [-3,1;0,1];
B = inv(A);
scale = max(max(abs(A)));
A_s = A / scale
B_s = B * scale

A * B
A_s * B_s

