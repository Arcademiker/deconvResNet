require 'nn';
require 'cunn';
local image = require 'image' --to view some images with recognised labels

opt = {}
opt.momentum = 0.9
opt.learningRate = 0.1
opt.learningRateDecay = 5.0e-6
opt.iterations = 15
opt.batchSize = 128
opt.layers = 15

local N = (opt.layers-10)/6


--local mnist = require 'mnist' --1:not CIFAR-10

--changed mnist loader:
--if (not paths.filep("cifar10torchsmall.zip")) then
--  os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
--  os.execute('unzip cifar10torchsmall.zip')
--end
--changeend(1)

--local train = mnist.traindataset() --2:not CIFAR-10
local train = torch.load('cifar10-train.t7') --changed(2)
--todo format
local Xt = train.data
local Yt = train.label
--local test = mnist.testdataset() --3:not CIFAR-10
local test = torch.load('cifar10-test.t7') --changed(3)
--todo format
local Xv = test.data
local Yv = test.label
Yt[Yt:eq(0)] = 10 --not CIFAR-10?
Yv[Yv:eq(0)] = 10 --not CIFAR-10?
--local train = require 'train' --changed look below
local model = require 'model'
--local net,ct = model.residual(N)
--print(net:__tostring__())


local sgd_config = {
      learningRate = opt.learningRate,
      learningRateDecay = opt.learningRateDecay,
      momentum = opt.momemtum
   }
--print('Number of convolutional layers .. '..#net:findModules('nn.SpatialConvolution'))





local train = {}
function train.sgd(Xt,Yt,Xv,Yv,K,sgd_config,batch)
    --local x,dx = net:getParameters()
    require 'optim' --view pretrained network?
    local batch = batch or 500
    print("datasetsize",Xt:size(1))
    --local Nt = Xt:size(1)
    --print('parameters size ..')
    --print(#x)
    --for k=1,K do
        --print(k,K)
        --print(os.date("%X", os.time()))
        --local lloss = 0
        --net:training() --view pretrained network?
        --for i = 1,Nt,batch do
            --xlua.progress(i/batch, Nt/batch)
            --dx:zero()
            --local j = math.min(i+batch-1,Nt)
            --local Xb = Xt[{{i,j}}]:cuda()
            --local Yb = Yt[{{i,j}}]:cuda()
            --local out = net:forward(Xb)
            --local loss = ct:forward(out,Yb)
            --local dout = ct:backward(out,Yb)
            --net:backward(Xb,dout)
            --dx:div(j-i+1)
            --function feval()
            --    return loss,dx
            --end
            --local ltmp,tmp = optim.sgd(feval,x,sgd_config)
            --print(loss)
            --lloss = lloss + loss
            --return loss
        --end
        local net2 = torch.load('net.t7') --added this line to view pretrained network
        --net = net2:double()
        --net2 = net2:cuda() --added this line to view pretrained network really required?
        --print('loss..'..lloss)
	net2 = net2:cuda()
        print('valid accuracy..'.. train.accuracy(Xv,Yv,net2,batch))
        --print('train accuracy..'.. train.accuracy(Xt,Yt,net2,batch))
        --torch.save('cpunet.t7',net)
    --end
end
function train.accuracy(Xv,Yv,net,batch)
    local classifier = {'airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
    --print(classifier)
    net:evaluate()
    local batch = batch or 512
    local Nv = Xv:size(1)
    local lloss = 0
    local secondcorrect = 0
    local firstcorrect = 0
    --local confusionM = torch.Tensor(10,10):zero()
    --confusionM = confusionM:cuda()
    --print(confusionM)
    --deconv ini:
    local deconvnet = nn.Sequential()
    for i=1,Nv,batch do
        xlua.progress(i/batch, Nv/batch)
        local j = math.min(i+batch-1,Nv)
        local Xb = Xv[{{i,j}}]:cuda()
        local Yb = Yv[{{i,j}}]:cuda()
        local out = net:forward(Xb)
        local tmp,YYb = out:max(2) 
        local lmax = i+batch-1 <= Nv and batch or (Nv%batch) --lazy evaluation
        for l=1, lmax do
          local trash, outsort = out[l]:sort()
          --print(l,classifier[Yv[l+i-1]],classifier[YYb[l][1]],classifier[outsort[9]])
          firstcorrect = Yv[l+i-1] == YYb[l][1] and firstcorrect + 1 or firstcorrect
          secondcorrect = Yv[l+i-1] == outsort[9] and secondcorrect + 1 or secondcorrect
          --confusionM[Yv[l+i-1]] = out[l]/out[l]:sum() + confusionM[Yv[l+i-1]]
          --confusionM[Yv[l+i-1]][YYb[l][1]] = confusionM[Yv[l+i-1]][YYb[l][1]] + 1 
        end
        --image.display(Xb)
        --deconv start:
        --todo netz bis zur letzten Spatial Convolution umkehren
        deconvnet:add(nn.SpatialFullConvolution(768,384,1,1,2,2))
        local out = deconvnet:forward(Xb)

        --todo falsch erkannte bilder ausgeben
        lloss = lloss + YYb:eq(Yb):sum()
    end
    --print(confusionM)
    print("top1 correct", firstcorrect)
    print("top2 correct", firstcorrect+secondcorrect)
    return (100*lloss/Nv)
end

--return train

train.sgd(Xt,Yt,Xv,Yv,opt.iterations,sgd_config,opt.batchSize)

--print(ex1.y)
--print(ex2.y)
--print(ex3.y)
--image.display(ex1.x)
--image.display(ex2.x)
--image.display(ex3.x)
