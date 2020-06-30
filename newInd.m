function resultado = newInd(IndAuto,mediana)


for i=1:length(IndAuto)
    if IndAuto(i,1)<=mediana
        IndAuto(i,1) = 0;
    else
        IndAuto(i,1) = 1;
    end
end

resultado = IndAuto;

end