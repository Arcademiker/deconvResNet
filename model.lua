require 'nn'

local model = {}
local res = require 'residual'


function model.residual(N,L)
    local N = N or 15
    local L = L or 1
    local half = true

    local net = nn.Sequential()
    --net:add(nn.Reshape(1,28,28))  --1:not CIFAR-10 3,32,32
    --res.convunit(net,1,64)        --2:not CIFAR-10 3 channels
    net:add(nn.Reshape(3,32,32))  --changed(1)
    --net:add(nn.SpatialConvolution(3,64,7,7, 2,2, 1,1)) --added by me
    --net:add(nn.ReLU(true))
    res.convunit(net,3,64*L)        --changed(2) --*3 added all layers
    net:add(nn.SpatialMaxPooling(2,2, 1,1)) --added by me (bei 64*64 pad von 2,2)
    res.rconvunitN(net,64*L,N)
    res.rconvunit2(net,64*L,half)
    res.rconvunitN(net,128*L,N)
    res.rconvunit2(net,128*L,half)
    res.rconvunitN(net,256*L,N)
    res.rconvunit2(net,256*L,half)
    net:add(nn.SpatialAveragePooling(2,2, 1,1)) --added by me
    cls = nn.Sequential()
    local wid = 3 --4
    cls:add(nn.Reshape(512*wid*wid*L))
    cls:add(nn.Linear(512*wid*wid*L,10))
    cls:add(nn.LogSoftMax())
    net:add(cls)
    local ct = nn.ClassNLLCriterion()

    require 'cunn';
    net = net:cuda()
    ct = ct:cuda()

    return net,ct
end
return model

----vorher:
--    net:add(nn.Reshape(3,32,32))  --changed(1)
--    res.convunit(net,3,64)        --changed(2)
--    res.rconvunitN(net,64,N)
--    res.rconvunit2(net,64,half)
--    res.rconvunitN(net,128,N)
--    res.rconvunit2(net,128,half)
--    res.rconvunitN(net,256,N)
--    res.rconvunit2(net,256,half)