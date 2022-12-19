%Studentas:	Valdas Mikenas		
%Kryptis (grupė):	Dirbtinio intelekto sistemos (DISfm-22)		


%% Daugiasluoksnis perceptronas 
%0. Nenaudodami MATLAB specializuotų funkcijų (newff, train, sim ir pan.), 
%1. Sukurkite daugiasluoksnį (dviejų sluoksnių) perceptronų tinklą ir jį apmokykite pasirinktai kreivei (sudarytai iš 20 atskaitų) aproksimuoti. 
%2. Tinklo apmokymui įgyvendinkite Backpropagation algoritmą. 
%3. Aktyvavimo funkcijas neuronų tinkle pasirinkite savo nuožiūra.
%4. Kreivę galite sukurti, pavyzdžiui, naudodami kelias trigonometrines funkcijas.


%Struktura:
%vienas iejimas
%vienas isejimas
%5 neuronai
%tiesine aktyvavimo f-cija
%mokymo algoritmas - Backpropagation

%Etapai
%% 1. Paruošiame pradinius duomenis mokymui x ir d
x=0.1:1/22:1;    % reiksmiu
n=length(x);     %20 reiksmiu

d=(1+0.6*sin(2*pi*x/0.7)+0.3*sin(2*pi*x))/2;  %f-cija
figure(1);
plot (x, d,'ko');
xlabel('x');
ylabel('d');
grid on;
hold on;

%% 2. Sugeneruojame pradines parametru reikšmes

%1 sluosknio parametrai (svoriai + bazės)
w11_1=rand(1); b1_1 =rand(1); %iejimas i aktyvavimo f-cija
w12_1=rand(1); b2_1 =rand(1);
w13_1=rand(1); b3_1 =rand(1);
w14_1=rand(1); b4_1 =rand(1);
w15_1=rand(1); b5_1 =rand(1);

%2 sluosknio parametrai (svoriai + bazė)
w11_2=rand(1); b1_2 =rand(1); 
w12_2=rand(1); 
w13_2=rand(1); 
w14_2=rand(1); 
w15_2=rand(1);

%% 3. Skaiciuojame tinklo atsaką v(pasverta suma) +aktyvavimo f-cija O(atsaka) ir gauname tinklo atsaka y (prognoze)   (1 - klase ar 0 - 2 klase) 
%sigmoides arba S-shap, vienas apribojimas intervale nuo 0...1, 

mok_zings=0.1  %mokymo zingsnis

for i = 1:10000
    for N = 1:20;

        %Apasvertos sumos 1 sluoksnyje    
        v1_1 = x(N)*w11_1 + b1_1; %imame po viena reiksme!
        v2_1 = x(N)*w12_1 + b2_1;
        v3_1 = x(N)*w13_1 + b3_1;
        v4_1 = x(N)*w14_1 + b4_1;
        v5_1 = x(N)*w15_1 + b5_1;
        
        %Aktyvavimo funkcija 1 sluoksnyje
        y1_1 = 1/(1+exp(-v1_1));
        y2_1 = 1/(1+exp(-v2_1));
        y3_1 = 1/(1+exp(-v3_1));
        y4_1 = 1/(1+exp(-v4_1));
        y5_1 = 1/(1+exp(-v5_1));
        
        %Pasverta suma 2 sluoksnyje arba isejimo sluoksnyje
        v1_2 = y1_1*w11_2+y2_1*w12_2+y3_1*w13_2+y4_1*w14_2+y5_1*w15_2 + b1_2;
        
        %NEteisine aktyvavimo funkcija
        y1_2 = tanh(v1_2);   %apskaiciuota

   
        y = y1_2;



  %% 4. Skaičiuojame klaidas e
        %e(index)=d(index)-y(index)
        %e(x)= atstumas tarp pradiniu reiksmiu ir gautu
        
        e = d(N) - y;
        
        
  %% 5. Skaičiuojame lokalius klaidos gradientus visiems neuronams (ar esame isejimo sluoksnyje ar paprastame)
        
  
  %Skaičiuojame klaidos gradientą išėjimo sl. neuronui
        delta1_2 = (1-(tanh(v1_2))^2)*e;
        
        %Skaičiuojame klaidos gradientą paslėptojo sl. neuronams
        % delta = e*aktyvacijos_fcijos_isvestine
        delta1_1 = y1_1*(1-y1_1)*delta1_2*w11_2;
        delta2_1 = y2_1*(1-y2_1)*delta1_2*w12_2;
        delta3_1 = y3_1*(1-y3_1)*delta1_2*w13_2;
        delta4_1 = y4_1*(1-y4_1)*delta1_2*w14_2;
        delta5_1 = y5_1*(1-y5_1)*delta1_2*w15_2;
        
        %Atnaujiname išėjimo sluoksnio svorius
        %w(i kuri neurona)(is kurio neurono)_(kelintas sluoksnis)
        
        w11_2 = w11_2 + mok_zings*delta1_2*y1_1;
        w12_2 = w12_2 + mok_zings*delta1_2*y2_1;
        w13_2 = w13_2 + mok_zings*delta1_2*y3_1;
        w14_2 = w14_2 + mok_zings*delta1_2*y4_1;
        w15_2 = w15_2 + mok_zings*delta1_2*y5_1;
        b1_2 = b1_2 + mok_zings*delta1_2;
        %2) S= - jei pasleptajame sluoksnyje
        
        %Atnaujiname paslėptojo sluoksnio svorius
        w11_1 = w11_1 + mok_zings*delta1_1*x(N);
        w12_1 = w12_1 + mok_zings*delta2_1*x(N);
        w13_1 = w13_1 + mok_zings*delta3_1*x(N);
        w14_1 = w14_1 + mok_zings*delta4_1*x(N);
        w15_1 = w15_1 + mok_zings*delta5_1*x(N);

        b1_1 = b1_1 + mok_zings*delta1_1;
        b2_1 = b2_1 + mok_zings*delta2_1;
        b3_1 = b3_1 + mok_zings*delta3_1;
        b4_1 = b4_1 + mok_zings*delta4_1;
        b5_1 = b5_1 + mok_zings*delta5_1;
      end
end

%% Laikome, kad po 10 tūkstančių iteracijų perėjus per visa įėjimo reikšmių
%% masyvą tinklas yra apmokytas.

% Po vieną tašką braižomas norimos atvaizduoti funkcijos grafikas ir
% apmokyto tinklo išėjimo vertės palyginimui    

% Skaičiuojame tinklo atsaką
x = [0.1:1/220:1];
Y = zeros(1, length(d));

for N=1:length(x)
       
        %Pirmojo sluoksnio neuronas
        v1_1 = x(N)*w11_1 + b1_1;
        v2_1 = x(N)*w12_1 + b2_1;
        v3_1 = x(N)*w13_1 + b3_1;
        v4_1 = x(N)*w14_1 + b4_1;
        v5_1 = x(N)*w15_1 + b5_1;

        %Aktyvavimo f-ijos  pritaikymas
        y1_1 = 1/(1+exp(-v1_1));
        y2_1 = 1/(1+exp(-v2_1));
        y3_1 = 1/(1+exp(-v3_1));
        y4_1 = 1/(1+exp(-v4_1));
        y5_1 = 1/(1+exp(-v5_1));

        %Antrojo sluoksnio neuronui
        v1_2 = y1_1*w11_2+y2_1*w12_2+y3_1*w13_2+y4_1*w14_2+y5_1*w15_2 + b1_2;

        %Aktyvavimo funkcijos pritaikymas 
        y1_2 = tanh(v1_2);
        y = y1_2;
        Y(N) = y;
end

hold on;
plot (x, Y, 'kx'); 
title('Originalus signalas ir apmokyto tinklo signalas');
legend('Originalus','OUT');
hold off;


%% Papildoma uzduotis su 2 iejimais ir 1 isejimu (nepadaryta)
%struktura:
%2 iejimai
%vienas isejimas
%5 neuronai