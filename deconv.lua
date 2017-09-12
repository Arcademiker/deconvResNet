nn = require 'nn'
torch = require 'torch'
cunn = require 'cunn'

----todo:
--pad str schecken bei erster conv
--pooling
--logsoftmax?

local deconv = {}

function deconv.net(N,L,W,imgsize)

    local net = nn.Sequential()


    local N = N or 15
    local L = L or 1
    imgsize = 32

   

    --deconv:
    --cls:add(nn.LogSoftMax()) --revers? wurde nie durchgeführt xD
    ----veralted:
    --local llinear1 = nn.Linear(10,512*wid*wid*L)
    --llinear1.weight = W.linear1.weight
    --llinear1.bias = W.linear1.bias
    --llinear1.gradWeight = W.linear1.gradWeight
    --llinear1.gradBias = W.linear1.gradBias
    --cls:add(llinear1)
    --local cls = nn.Sequential()
    --local wid = 37 --4
    --cls:add(nn.Reshape(256*L,wid,wid))

    --net:add(cls)
    --net:add(nn.SpatialAveragePooling(2,2, 1,1)) --added by me
    --net:add(nn.SpatialMaxUnpooling(W.pool2))
    local dfc1 = nn.SpatialFullConvolution(2,512,1,1,1,1,0,0)
    dfc1.weight = W.fc1.weight
    dfc1.bias = dfc1.bias:fill(0)
    dfc1.gradWeight = W.fc1.gradWeight
    dfc1.gradBias = dfc1.gradBias:fill(0)
    net:add(dfc1)

    local Tconv1 = {}
    local Tbatch1 = {}
    local Tconv2 = {}
    local Tbatch2 = {}
    --local TconvT = {}
    local conv1
    local batch1
    local conv2
    local batch2
    --local convT
    local fin
    local pad = 1
    local gstr = 0
    local fsize = 4
    local adj = 0
    for l=1,3 do                         -- !!!
        if l==1 then
            fin = 256*L
            conv1  = W.conv12
            batch1 = W.batch12
            conv2  = W.conv13
            batch2 = W.batch13
        elseif l==2 then
            fin = 128*L
            conv1  = W.conv8
            batch1 = W.batch8
            conv2  = W.conv9
            batch2 = W.batch9
        else
            fin = 64*L
            conv1  = W.conv4
            batch1 = W.batch4
            conv2  = W.conv5
            batch2 = W.batch5
        end

        --local function lrconvunit2(net,fin,half,conv1,batch1,conv2,batch2,convT)

        --local function lconvunit2(net,fin,half,conv1,batch1,conv2,batch2)

        --local function lconvunit2(net,fin,half,conv1,batch1,conv2,batch2)



        --end

        --local function lconvunit31(net,fin,half,str,nobatch,conv1,batch1)
        local str = 1 --+gstr --1
        local fout = fin

        --end

        --local function lconvunit(net,fin,fout,fsize,str,pad,nobatch,conv1,batch1)
        local nobatch = false


        if(nobatch==true) then net:add(nn.ReLU(true)) end

        local lbatch2 = nn.SpatialBatchNormalization(2*fout)
        lbatch2.weight = batch2.weight
        lbatch2.bias = batch2.bias
        lbatch2.gradWeight = batch2.gradWeight
        lbatch2.gradBias = batch2.gradBias
        net:add(lbatch2)

        local lconv2 = nn.SpatialFullConvolution(2*fout,2*fin,fsize,fsize,str,str,pad,pad,adj,adj)
        lconv2.weight = conv2.weight
        lconv2.bias = conv2.bias
        lconv2.gradWeight = conv2.gradWeight
        lconv2.gradBias = conv2.gradBias
        net:add(lconv2)
        --end

        --local function lconvunit31(net,fin,half,str,nobatch,conv1,batch1)

        nobatch = true


        fout = fin
        str = 1 --2+gstr --1


        --end

        --local function lconvunit(net,fin,fout,fsize,str,pad,nobatch,conv1,batch1)



        if(nobatch==true) then net:add(nn.ReLU(true)) end

        local lbatch1 = nn.SpatialBatchNormalization(2*fout)
        lbatch1.weight = batch1.weight
        lbatch1.bias = batch1.bias
        lbatch1.gradWeight = batch1.gradWeight
        lbatch1.gradBias = batch1.gradBias
        net:add(lbatch1)

        local lconv1 = nn.SpatialFullConvolution(2*fout,fin,fsize,fsize,str,str,pad,pad,adj,adj)
        lconv1.weight = conv1.weight
        --lconv1.bias = conv1.bias
        lconv1.gradWeight = conv1.gradWeight
        --lconv1.gradBias = conv1.gradBias
        net:add(lconv1)
        --end

        --end

        --local function lresUnit(net, unit, fin,convT)
        --net = net or nn.Sequential()
        --net:add(unit)
        --end

        --end

        --local function lrconvunitN(net,fin,N,Tconv1,Tbatch1,Tconv2,Tbatch2,TconvT)
        if l==1 then
            fin = 256*L
            Tconv1  = W.Tconv10
            Tbatch1 = W.Tbatch10
            Tconv2  = W.Tconv11
            Tbatch2 = W.Tbatch11
        elseif l==2 then
            fin = 128*L
            Tconv1  = W.Tconv6
            Tbatch1 = W.Tbatch6
            Tconv2  = W.Tconv7
            Tbatch2 = W.Tbatch7
        else
            fin = 64*L
            Tconv1  = W.Tconv2
            Tbatch1 = W.Tbatch2
            Tconv2  = W.Tconv3
            Tbatch2 = W.Tbatch3
        end
        for i=N,1,-1 do
            conv1 = Tconv1[i]
            batch1 = Tbatch1[i]
            conv2 = Tconv2[i]
            batch2 = Tbatch2[i]

            --local function lrconvunit2(net,fin,half,conv1,batch1,conv2,batch2,convT)

            --local function lconvunit2(net,fin,half,conv1,batch1,conv2,batch2)

            --local function lconvunit2(net,fin,half,conv1,batch1,conv2,batch2)


            --end

            --local function lconvunit31(net,fin,half,str,nobatch,conv1,batch1)
            str = 1 --+gstr --1
            fout = fin

            --end

            --local function lconvunit(net,fin,fout,fsize,str,pad,nobatch,conv1,batch1)
            nobatch = false



            if(nobatch==true) then net:add(nn.ReLU(true)) end

            local lbatch2 = nn.SpatialBatchNormalization(fout)
            lbatch2.weight = batch2.weight
            lbatch2.bias = batch2.bias
            lbatch2.gradWeight = batch2.gradWeight
            lbatch2.gradBias = batch2.gradBias
            net:add(lbatch2)

            local lconv2 = nn.SpatialFullConvolution(fout,fin,fsize,fsize,str,str,pad,pad,adj,adj)
            lconv2.weight = conv2.weight
            lconv2.bias = conv2.bias
            lconv2.gradWeight = conv2.gradWeight
            lconv2.gradBias = conv2.gradBias
            net:add(lconv2)
            --end

            --local function lconvunit31(net,fin,half,str,nobatch,conv1,batch1)
            nobatch = true

            str = 1 --+gstr --1
            fout = fin


            --end

            --local function lconvunit(net,fin,fout,fsize,str,pad,nobatch,conv1,batch1)



            if(nobatch==true) then net:add(nn.ReLU(true)) end

            local lbatch1 = nn.SpatialBatchNormalization(fout)
            lbatch1.weight = batch1.weight
            lbatch1.bias = batch1.bias
            lbatch1.gradWeight = batch1.gradWeight
            lbatch1.gradBias = batch1.gradBias
            net:add(lbatch1)

            local lconv1 = nn.SpatialFullConvolution(fout,fin,fsize,fsize,str,str,pad,pad,adj,adj)
            lconv1.weight = conv1.weight
            lconv1.bias = conv1.bias
            lconv1.gradWeight = conv1.gradWeight
            lconv1.gradBias = conv1.gradBias
            net:add(lconv1)
            --end

            --end

            --local function lresUnit(net, unit, fin,convT)
            --net = net or nn.Sequential()
            --net:add(unit)
            --end

            --end

        end
        --end
    end

    --net:add(nn.SpatialMaxUnpooling(W.pool1))
    --net:add(nn.SpatialMaxPooling(2,2, 1,1)) --added by me (bei 64*64 pad von 2,2)
    net:add(nn.ReLU(true)) --new
    --local function lconvunit(net,fin,fout,fsize,str,pad,nobatch,conv1,batch1)
    fin = 3
    local fout = 64*L
    local str = 1 --+gstr --1


    local conv1 = W.conv1
    local batch1 = W.batch1
    local nobatch = false
    if(nobatch==true) then net:add(nn.ReLU(true)) end

    local lbatch1 = nn.SpatialBatchNormalization(fout)
    lbatch1.weight = batch1.weight
    lbatch1.bias = batch1.bias
    lbatch1.gradWeight = batch1.gradWeight
    lbatch1.gradBias = batch1.gradBias
    net:add(lbatch1)

    local lconv1 = nn.SpatialFullConvolution(fout,fin,fsize,fsize,str,str,pad,pad,adj,adj)
    lconv1.weight = conv1.weight
    --lconv1.bias = conv1.bias
    lconv1.gradWeight = conv1.gradWeight
    --lconv1.gradBias = conv1.gradBias
    net:add(lconv1)
    --net:add(nn.ReLU(true))
    --net:add(nn.SpatialConvolution(3,64,7,7, 2,2, 1,1)) --added by me
    --end
    --net:add(nn.Reshape(3,32,32))  --changed(1)
    net:add(nn.Reshape(3,imgsize,imgsize))  --changed(1)

    local ct = nn.ClassNLLCriterion() --??
    net = net:cuda()
    ct = ct:cuda()

    return net,ct

end

return deconv



