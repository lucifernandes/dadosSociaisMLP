%----------Prepara��o dos dados----------

%C�lculo dos �ndices de cada amostra, onde InAuto ser� a coluna
%correspondente a esses valores
IndAuto = mean(novabase1(:,47:63),2);

%Mediana dos valores de �ndice de Autoritarismo
mediana = median(IndAuto);

%Base com modifica��es aplicadas

t = newInd(IndAuto,mediana).';
x = [novabase1(:,1:46) novabase1(:,64:78)].'; %Entradas (retirei as respostas do 4� bloco do question�rio)

%----------C�digo----------

net = patternnet(2);
net = train(net,x,t);
view(net)
y = net(x);
perf = perform(net,t,y);
classes = vec2ind(y);