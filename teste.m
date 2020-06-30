close all;
clear all;
clc;
disp('Programa MLP para XOR 2 entradas');
% Entradas
X = [ -1 , 1 , -1 , 1 ;
 -1 , -1 , 1 , 1 ];
% Saida desejada: funcao logica XOR
Yd =[ -1 , 1 , 1 , -1 ];
% Quantidade de neuronios na camada escondida
neuronios_camada_escondida = 2;
disp('Criando a rede MLP...');
net = feedforwardnet(neuronios_camada_escondida);
disp('Configurando a rede...');
net = configure(net,X,Yd);
% divisao dos dados entre treinamento, teste, validacao
net.divideParam.trainRatio = 1; % training set [%]
net.divideParam.valRatio = 0; % validation set [%]
net.divideParam.testRatio = 0; % test set [%]
% Ajusta parametros para treinamento:
% metodos de treinamento: backpropagation (traingd), backpropagation com
% momentum (traingdm), Levenberg-Marquardt (trainlm), RPROP (trainrp)
net.trainFcn = 'traingd';
% funcao a ser minimizada: mse, mae, sse
net.performFcn = 'mse'; % erro medio quadratico
% numero maximo de epocas para treinamento
net.trainParam.epochs = 5000;
% taxa de aprendizado
net.trainParam.lr = 0.01;
% taxa de momento
net.trainParam.mc = 0;
% erro maximo permitido
net.trainParam.goal = 0.01;
% Ajusta funcao de ativacao de cada camada da rede: tansig, logsig
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
% Inicializa a rede neural com valores de pesos e bias aleatorios
disp('Inicializando a rede neural....');
net = init(net);
% Treina a rede neural
disp('Treinando a rede neural...');
[net, tr] = train(net,X,Yd);
% Plota o grafico do erro durante treinamento
plotperform(tr);
% Simula a rede neural
disp('Simulando a rede neural treinada...');
Ysaida = sim(net,X);
disp('Resultado: ');
disp(Ysaida);
% Calcula o desempenho obtido
disp('Calculando o erro da rede neural...');
perf = perform(net,Yd,Ysaida);
disp('Erro: ');
disp(perf);
% Obtem os valores de pesos e bias em um unico vetor
wb = getwb(net);
% Separa os valores dos pesos e bias
[b,IW,LW] = separatewb(net,wb);
% Converte de celula para vetor
b = cell2mat(b);
disp('Bias de todas camadas =');
disp(b);
% Converte de celula para vetor
Wescondida = cell2mat(IW);
disp('Pesos da camada escondida =');
disp(Wescondida);
% Converte de celula para vetor
Wsaida = cell2mat(LW);
disp('Pesos da camada de saida =');
disp(Wsaida);