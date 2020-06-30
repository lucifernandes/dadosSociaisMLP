function divisao(base)

k=0;
testagem = [];
n = 1757;
    for i=1:167

        k = randi([1, n]);
        testagem = base(k,:);
        base(k,:) = [];                                                  
        n = n-1;

    end

end