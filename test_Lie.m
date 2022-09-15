Ez = [0,-1,0;...
    1,0,0;...
    0,0,0,];
Ey = [0,0,1;...
    0,0,0;...
    -1,0,0,];
Ex = [0,0,0;...
    0,0,-1;...
    0,1,0,];
I = [1,0,0;...
    0,1,0;...
    0,0,1];
u = [0,0,1];
theta = 3.1415926/2;
Ux = Ex*u(1)+Ey*u(2)+Ez*u(3);

R = I + Ux*sin(theta)+Ux*Ux*(1-cos(theta));
R*[1,2,0]'

theta2 = 3.13/2;
Ux = Ex*u(1)+Ey*u(2)+Ez*u(3);
R2 = I + Ux*sin(theta2)+Ux*Ux*(1-cos(theta2));
dR = (R2-R)/(-0.01)

theta3 = 0.001;
Ux = Ex*u(1)+Ey*u(2)+Ez*u(3);
R3 = I + Ux*sin(theta3)+Ux*Ux*(1-cos(theta3));
dR = (R3-eye(3))/(0.001)