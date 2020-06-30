clc;close all;clear all;
cd('H:\Articles\SoftwareX ITC\ITC-master'); % Files Location

ITC=xlsread('Data7.xls');
try
    ITC2=xlsread('Data1.xls');
catch
end

Delete_=1;                              %Number of initial points to delete

Clipid=15*10^-3;                        %Concentration of the Lipid [M]
Cdrug=100*10^-6;                        %Concentration of the Drug in the cell [M]
Vw=200*10^-6;                           %Volume of the Cell [L]

vL=(1398*10^-27)*(6.0221409*10^23);     %Vesicles Volume
aL=(7e-19)*(6.0221409e23);              %Vesicles Area
aD=(3e-19)*(6.0221409e23);              %Drug Area

R=8.314459848;                          %J/(mol*Kelvin)
T=298.15;                               %Kelvin
F=96485.336521;                         %C/mol
eps0=8.85*10^-12;                       %Farads/m
eps=78.4;                               %Farads/m
SaltC=0.15;                             %moles/L Salt concentration
zdrug= 1;                               %Charge of drug

number_of_injection=size(ITC,1);
Vinj=ITC([1:1:number_of_injection-1],2)*10^-6;
x=[1:1:number_of_injection-(Delete_+1)]';
heat=ITC([(Delete_+1):1:number_of_injection-1],1);

number_of_injection=size(ITC,1);
Vinj=ITC([1:1:number_of_injection-1],2)*10^-6;
x=[1:1:number_of_injection-(Delete_+1)]';
heat=ITC([(Delete_+1):1:number_of_injection-1],1);
Vinj=Vinj(end-1);

try
    number_of_injection2=size(ITC2,1);
    heat2=ITC2([(Delete_+1):1:number_of_injection2-1],1);
    if size(heat,1)>size(heat2,1)
        A=zeros(1,abs(size(heat,1)-size(heat2,1)));
        heat2=[heat2;A'];
    elseif size(heat,1)<size(heat2,1)
        heat2=heat2(1:size(heat,1))
    end
    heat=heat-heat2;
catch
end

Kp=1;dH=heat(1)*T;
dHeat=[0 heat(1)*inf];
for i=1:1:1000
    s = fitoptions('Method','NonlinearLeastSquares','Lower',[min(dHeat) 10^-20],...
    'Upper',[max(dHeat) inf],'Startpoint',[dH Kp],'MaxFunEvals',1e200);
    f = fittype('(Vw*Cdrug)*Vw*Kp*Vinj*Clipid*aL*dH*(10^6)/(Vw+(x-0.5)*Vinj*(1+Clipid*aL*Kp))^2','dependent',{'heat'},'independent',{'x'},...
    'problem',{'Cdrug','Vw','Vinj','Clipid','aL'},'coefficients',{'dH','Kp'}, 'options',s);
    [h gof]= fit(x,heat,f,'problem',{Cdrug,Vw,Vinj,Clipid,aL});
    coeff=coeffvalues(h);
    dH2=coeff(1);
    Kp2=coeff(2);
    dG2=-1.9858775*298.15*log(Kp2*(aL/vL));
    TdS2=dH2-dG2;

    if abs(Kp2-Kp)+abs(dH2-dH)<=0
        break
    else
        Kp=Kp2;
        dH=dH2;
    end
end

Results=[Kp2*(aL/vL),dH2,dG2,TdS2,gof.rsquare,gof.sse];

k=((8000*R*T*eps*eps0*SaltC)^.5);
beta=zdrug*F/(R*T);
const1=Clipid*aL*Vinj;
const2=const1*k/(zdrug*F);

if zdrug~=0
    for i=1:1:1000
        Phi=[];
        heat=heat/10^6;
        for j=1:1:size(x,1)
            A=sum(heat(1:j))/(x(j)*const2*dH);
            B=dH*x(j)*const1/(sum(heat(1:j))*aD+dH*x(j)*const1);
            Phi_=(2/beta)*asinh(A*B);
            Phi=[Phi;Phi_];
        end
        heat=heat*10^6;

        s = fitoptions('Method','NonlinearLeastSquares','Lower',[min(dHeat) 10^-20],...
        'Upper',[max(dHeat) inf],'Startpoint',[dH Kp],'MaxFunEvals',1e200);
        f = fittype('10^6*((dH*(Vw*Cdrug)*x*const1*Kp*exp(-beta*Phi)/(Vw+x*Vinj+x*const1*Kp*exp(-beta*Phi)))-(dH*(Vw*Cdrug)*(x-1)*const1*Kp*exp(-beta*Phi)/(Vw+(x-1)*Vinj+(x-1)*const1*Kp*exp(-beta*Phi))))','dependent',{'heat'},'independent',{'x','Phi'},...
        'problem',{'Cdrug','const1','beta','Vw','Vinj'},'coefficients',{'dH','Kp'}, 'options',s);
        [h gof]= fit([x,Phi],heat,f,'problem',{Cdrug,const1,beta,Vw,Vinj});
        coeff=coeffvalues(h);
        dH2=coeff(1);
        Kp2=coeff(2);
        dG2=-1.9858775*298.15*log(Kp2*(aL/vL));
        TdS2=dH2-dG2;

        if abs(Kp2-Kp)+abs(dH2-dH)<=0
            break
        else
            Kp=Kp2;
            dH=dH2;
        end
    end
    Results=[Results;[Kp2*(aL/vL),dH2,dG2,TdS2,gof.rsquare,gof.sse]];
    subplot(2,1,1);plot(x,h(x,Phi),'r');legend('off');
    hold on;plot(x,heat,'.')
    xlabel('Injection Number');ylabel('\mu cal');
    subplot(2,1,2);plot(x,Phi*1000,'+');legend('off');
    xlabel('Injection Number');ylabel('\Phi mV');
end

if zdrug==0
    Results=[Results;[Kp2*(aL/vL),dH2,dG2,TdS2,gof.rsquare,gof.sse]];
    subplot(2,1,1);plot(x,h(x),'r');legend('off');
    hold on;plot(x,heat,'.')
    xlabel('Injection Number');ylabel('\mu cal');
end

Resultstxt={'Kp','deltaH','deltaG','t_delta_s','rsquare','sse'};
Results= num2cell(Results);
Results=cat(1,Resultstxt,Results);
openvar Results