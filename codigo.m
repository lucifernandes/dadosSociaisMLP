%----------Preparação dos dados----------

%Cálculo dos índices de cada amostra, onde InAuto será a coluna
%correspondente a esses valores
IndAuto = mean(novabase1(:,47:63),2);

%Mediana dos valores de índice de Autoritarismo
mediana = median(IndAuto);

%Base com modificações aplicadas

t = newInd(IndAuto,mediana).';
x = [novabase1(:,1:46) novabase1(:,64:78)].'; %Entradas (retirei as respostas do 4º bloco do questionário)

%----------Código----------

net = patternnet(2);
net = train(net,x,t);
view(net)
y = net(x);
perf = perform(net,t,y);
classes = vec2ind(y);