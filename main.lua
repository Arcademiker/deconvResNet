--todo auf CIFAR-10 ändern


require 'nn';
require 'cunn';
require 'image'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST training using Residual Neural Networks')
cmd:text('Example:')
cmd:text('$> th main.lua --layers 50 --batchSize 128 --iterations 50 --large 3')
cmd:text('Options:')
cmd:option('--momentum', 0.9, 'momemtum during SGD')
cmd:option('--learningRate', 0.1, 'learning rate at t=0') --/10 when error plateaus
--cmd:option('--learningRateDecay', 5.0e-6, 'learning rate decay')
--cmd:option('--iterations', 30, 'number of iterations to run')
--cmd:option('--batchSize', 128, 'batch size(adjust to fit in GPU)')
--cmd:option('--layers', 100, 'approx num of layers to train')
cmd:option('--learningRateDecay', 0.0001, 'learning rate decay')
cmd:option('--iterations', 2, 'number of iterations to run')
cmd:option('--batchSize', 128, 'batch size(adjust to fit in GPU)')
cmd:option('--layers', 10, 'approx num of layers to train')
cmd:option('--verbose', 1, '0 = dont print xlua commands and test at the end')
cmd:option('--large', 1, 'number of neurons per layer')
cmd:option('--pre', 0, '1 = use preprosessing')


cmd:text()
opt = cmd:parse(arg or {})
if not opt.silent then
    print(opt)
end

opt.momentum = tonumber(opt.momentum)
opt.learningRate = tonumber(opt.learningRate)
opt.learningRateDecay = tonumber(opt.learningRateDecay)
opt.iterations = tonumber(opt.iterations)
opt.batchSize = tonumber(opt.batchSize)
opt.layers = tonumber(opt.layers)
opt.verbose = tonumber(opt.verbose)
opt.large = tonumber(opt.large)
opt.pre = tonumber(opt.pre)

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
print("trainset",train)
--todo format
local Xt = train.data
local Yt = train.label
--local test = mnist.testdataset() --3:not CIFAR-10
local test = torch.load('cifar10-test.t7') --changed(3)
print("testset",test)
--todo format
local Xv = test.data
local Yv = test.label
--???:
Yt[Yt:eq(0)] = 10 --not CIFAR-10?
Yv[Yv:eq(0)] = 10 --not CIFAR-10?

if (opt.pre==1) then
    -- preprocess trainSet
    Xt = Xt:double()
    local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
    for i = 1,Xt:size(1) do
        xlua.progress(i, Xt:size(1))
        -- rgb -> yuv
        local rgb = Xt[i]
        local yuv = image.rgb2yuv(rgb)
        -- normalize y locally:
        yuv[1] = normalization(yuv[{{1}}])
        Xt[i] = yuv

    end
    -- normalize u globally:
    local mean_u = Xt:select(2,2):mean()
    local std_u = Xt:select(2,2):std()
    Xt:select(2,2):add(-mean_u)
    Xt:select(2,2):div(std_u)
    -- normalize v globally:
    local mean_v = Xt:select(2,3):mean()
    local std_v = Xt:select(2,3):std()
    Xt:select(2,3):add(-mean_v)
    Xt:select(2,3):div(std_v)

    -- preprocess testSet
    Xv = Xv:double()
    for i = 1,Xv:size(1) do
        xlua.progress(i, Xv:size(1))
        -- rgb -> yuv
        local rgb = Xv[i]
        local yuv = image.rgb2yuv(rgb)
        -- normalize y locally:
        yuv[{1}] = normalization(yuv[{{1}}])
        Xv[i] = yuv
    end
    -- normalize u globally:
    Xv:select(2,2):add(-mean_u)
    Xv:select(2,2):div(std_u)
    -- normalize v globally:
    Xv:select(2,3):add(-mean_v)
    Xv:select(2,3):div(std_v)
end
--[[
--normalising: (make your data to have a mean of 0.0 and standard-deviation of 1.0)
    local mean = {} -- store the mean, to normalize the test set in the future
    local stdv  = {} -- store the standard-deviation for the future
    for i=1,3 do -- over each image channel
        mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
        print('Channel ' .. i .. ', Mean: ' .. mean[i])
        trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
        stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
        print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
        trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
    end
]]

--start the NN
local train = require 'train'
local model = require 'model'
local net,ct = model.residual(N,opt.large)
print(net:__tostring__())
local sgd_config = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    momentum = opt.momemtum
}
print('Number of convolutional layers .. '..#net:findModules('nn.SpatialConvolution'))
if (opt.verbose ~= 0) then
    train.sgd(net,ct,Xt,Yt,Xv,Yv,opt.iterations,sgd_config,opt.batchSize)
else
    train.sgdv(net,ct,Xt,Yt,Xv,Yv,opt.iterations,sgd_config,opt.batchSize)
end
