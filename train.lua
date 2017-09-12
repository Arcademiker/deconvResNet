local train = {}
function train.sgd(net,ct,Xt,Yt,Xv,Yv,K,sgd_config,batch)
    local x,dx = net:getParameters()
    local optim = require 'optim'
    local batch = batch or 500
    local Nt = Xt:size(1)
    print("datasetsize",Xt:size(1))
    print('parameters size ..')
    print(#x)
    local pretestaccuracy = 0
    local testaccuracy = 0
    local platteau = 1
    local flat = 0
    for k=1,K do
        print(k,K)
        print(os.date("%X", os.time()))
        local lloss = 0
        net:training()
        for i = 1,Nt,batch do
            xlua.progress(i/batch, Nt/batch)
            dx:zero()
            local j = math.min(i+batch-1,Nt)
            local Xb = Xt[{{i,j}}]:cuda()
            local Yb = Yt[{{i,j}}]:cuda()
            local out = net:forward(Xb)
            local loss = ct:forward(out,Yb)
            local dout = ct:backward(out,Yb)
            net:backward(Xb,dout)
            dx:div(j-i+1)
            function feval()
                return loss,dx
            end
            local ltmp,tmp = optim.sgd(feval,x,sgd_config)
            --print(loss)
            lloss = lloss + loss
            --return loss
        end
        print('loss..'..lloss)
        testaccuracy = train.accuracy(Xv,Yv,net,batch)
        print('valid accuracy..'.. testaccuracy)
        print('train accuracy..'.. train.accuracy(Xt,Yt,net,batch))
        if(pretestaccuracy+0.1 > testaccuracy) then
            if(platteau<flat) then
                sgd_config.learningRate = sgd_config.learningRate/10
                print(sgd_config.learningRate)
                platteau = platteau*2
                flat = -1
            end
            flat = flat+1
        end
        pretestaccuracy = testaccuracy
        if (k%10==0) then
            torch.save('net.t7',net)
        end
    end
    print("saving")
    torch.save('net.t7',net)
    net = net:double()
    torch.save('cpunet3.t7',net)
end
function train.accuracy(Xv,Yv,net,batch)
    net:evaluate()
    local batch = batch or 512
    local Nv = Xv:size(1)
    local lloss = 0
    for i =1,Nv,batch do
        xlua.progress(i/batch, Nv/batch)
        local j = math.min(i+batch-1,Nv)
        local Xb = Xv[{{i,j}}]:cuda()
        local Yb = Yv[{{i,j}}]:cuda()
        local out = net:forward(Xb)
        local tmp,YYb = out:max(2)
        lloss = lloss + YYb:eq(Yb):sum() --TE: cutorch 1.0-0
    end
    return (100*lloss/Nv)
end

--ohne ladebalken für pipe in datei
function train.sgdv(net,ct,Xt,Yt,Xv,Yv,K,sgd_config,batch)
    local x,dx = net:getParameters()
    require 'optim'
    local batch = batch or 500
    local Nt = Xt:size(1)
    print("datasetsize",Xt:size(1))
    print('parameters size ..')
    print(#x)
    local pretrainaccuracy = 0
    local trainaccuracy = 0
    local platteau = 1
    local flat = 0
    for k=1,K do
        print(k,K)
        print(os.date("%X", os.time()))
        local lloss = 0
        net:training()
        for i = 1,Nt,batch do
            --xlua.progress(i/batch, Nt/batch)
            dx:zero()
            local j = math.min(i+batch-1,Nt)
            local Xb = Xt[{{i,j}}]:cuda()
            local Yb = Yt[{{i,j}}]:cuda()
            local out = net:forward(Xb)
            local loss = ct:forward(out,Yb)
            local dout = ct:backward(out,Yb)
            net:backward(Xb,dout)
            dx:div(j-i+1)
            function feval()
                return loss,dx
            end
            local ltmp,tmp = optim.sgd(feval,x,sgd_config)
            --print(loss)
            lloss = lloss + loss
            --return loss
        end
        print('loss..'..lloss)
        trainaccuracy = train.accuracy(Xt,Yt,net,batch)
        print('train accuracy..'.. trainaccuracy)
        if(pretrainaccuracy+0.1 > trainaccuracy) then
            if(platteau<flat) then
                sgd_config.learningRate = sgd_config.learningRate/10
                print(sgd_config.learningRate)
                platteau = platteau*2
                flat = -1
            end
            flat = flat+1
        end
        pretrainaccuracy = trainaccuracy
        if (k%10==0) then
            torch.save('net.t7',net)
        end
    end
    print('valid accuracy..'.. train.accuracy(Xv,Yv,net,batch))
    print("saving")
    torch.save('net.t7',net)
    net = net:double()
    torch.save('cpunet3.t7',net)
end
function train.accuracyv(Xv,Yv,net,batch)
    net:evaluate()
    local batch = batch or 512
    local Nv = Xv:size(1)
    local lloss = 0
    for i =1,Nv,batch do
        --xlua.progress(i/batch, Nv/batch)
        local j = math.min(i+batch-1,Nv)
        local Xb = Xv[{{i,j}}]:cuda()
        local Yb = Yv[{{i,j}}]:cuda()
        local out = net:forward(Xb)
        local tmp,YYb = out:max(2)
        lloss = lloss + YYb:eq(Yb):sum() --TE: cutorch 1.0-0
    end
    return (100*lloss/Nv)
end

return train