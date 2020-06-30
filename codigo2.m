%---------- Coleta e preparação dos dados ----------
%Cálculo dos índices de cada amostra, onde InAuto será a coluna
%correspondente a esses valores

%base = basemodificada/norm(basemodificada);
base = basemodificada;

%[trainInd,valInd,testInd] = dividerand(1757,0.6,0.2,0.2);

IndAuto = mean(base(:,69:85),2);

mediana = median(IndAuto); %Mediana dos valores de índice de Autoritarismo

t = newInd(IndAuto,mediana).'; 
% yTrain = t(trainInd, :).';  %Targets
% yVal = t(valInd, :).';
% yTest = t(testInd, :).';

x = base(:,2:68).'; %Entradas 
% xTrain = base(trainInd,2:68).'; %Entradas 
% xVal = base(valInd,2:68).';
% xTest = base(testInd,2:68).';

%---------- Criação da rede ------------------------

camada_escondida = 67; %Camadas escondidas e quantidades de neurônios correspondentes
disp('Criando a rede MLP...');
net = feedforwardnet(camada_escondida);
%net = newff(minmax(x),[63 63 1], {'tansig' 'tansig' 'tansig'});

%---------- Configuração da rede -------------------

disp('Configurando a rede...');
net = configure(net,x,t);

%---------- Treinamento da rede --------------------

net.performFcn = 'mse'; % erro médio quadrático
%net.performFcn = 'mae'; % erro médio absoluto
%net.performFcn = 'crossentropy'; %Cross-entropy performance
%net.performFcn = 'msesparse'; %Mean squared error performance function with L2 weight and sparsity regularizers.

net.trainFcn = 'traingd'; %Gradient descent backpropagation
%net.trainFcn = 'traingda'; %Gradient descent with adaptive learning rate backpropagation
%net.trainFcn = 'traingdm'; %Gradient descent with momentum backpropagation
%net.trainFcn = 'traingdx'; %Gradient descent with momentum and adaptive learning rate backpropagation


net.trainParam.epochs = 5000; %épocas
net.trainParam.lr = 0.01; % taxa de aprendizado
net.trainParam.goal = 0; % erro maximo permitido
%net.trainParam.showCommandLine = true;
net.trainParam.max_fail = 100;
%net.trainParam.min_grad = 0;

%Função de ativação de cada camada da rede
net.layers{1}.transferFcn = 'tansig'; %Hyperbolic tangent sigmoid transfer function
net.layers{2}.transferFcn = 'tansig';
%net.layers{3}.transferFcn = 'tansig';
%net.layers{4}.transferFcn = 'tansig';
%net.layers{3}.transferFcn = 'tansig'; 
%net.layers{3}.transferFcn = 'logsig';
%net.layers{2}.transferFcn = 'purelin';

%---------- Inicialização de pesos e biases --------
disp('Inicializando a rede neural....');
net = init(net);
%net.plotFcns = {'plotconfusion'};
disp('Treinando a rede neural...');
[net, tr] = train(net,x,t);

%---------- Validação da rede ----------------------

%c = cvpartition(yVal,'KFold',10);






% Gráfico do erro durante treinamento
plotperform(tr);

% Simulação da rede neural
disp('Simulando a rede neural treinada...');
Ysaida = sim(net,x); 

% Cálculo o desempenho obtido
disp('Calculando o erro da rede neural...');
perf = perform(net,t,Ysaida);
disp('Erro: ');
disp(perf);
% 
% a=0; %Quantidade de acertos
% Ysaida = round(Ysaida); %Arredondamento dos valores
% for i=1:1757 %Total de amostras
%     if Ysaida(1,i)==t(1,i)
%         a = a + 1;
%     end
% end
% ac = a/1757;
% disp('Taxa de Acertos (%): ')
% disp(ac*100)
% 
%  



