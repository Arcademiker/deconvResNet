require 'nn';
require 'cunn';
require 'torch';
local image = require 'image' --to view some images with recognised labels

local decode = {}
function decode.sgd(net,dnet,W,ct,Xv,Yv,batch,imgsize,sgd_config)
    local x,dx = net:getParameters()
    local batch = batch or 500
    local Nt = Xv:size(1)
    print("datasetsize",Xv:size(1))
    print('parameters size ..')
    print(#x)
    print(os.date("%X", os.time()))

    decode.accuracy(Xv,Yv,net,dnet,W,batch,imgsize)
    --print("saving")
end
function decode.accuracy(Xv,Yv,net,dnet,W,batch,imgsize,sgd_config)
    local classifier = {'airplane', 'automobile', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
    net:evaluate()
    dnet:evaluate()
    imgsize = 32
    local xt,dxx = dnet:getParameters()
    local batch = batch or 512
    local Nv = 128 -- Xv:size(1)
    --local lloss = 0
    local list = {}
    --[[
    local fc = nn.Sequential()
    local testnet = nn.Sequential()
    testnet:add(nn.Reshape(256,imgsize,imgsize))
    --testnet:add(nn.SpatialFullConvolution(256,1,1,1,1,1,2,2))
    testnet = testnet:cuda()
    testnet:evaluate()
    ----testend
    fc:add(W.linear1)
    fc:add(nn.LogSoftMax()) --???nötig
    fc = fc:cuda()
    fc:evaluate()
    local A = W.linear1.weight
    local B = W.linear1.bias
    local me = torch.mean(Xv:double())/torch.mean(A)
    --]]
    local fcc = nn.Sequential()
    fcc:add(nn.Sum(3, 3))
    fcc:add(nn.Sum(2, 2))
    fcc:add(nn.Reshape(2))
    fcc:add(nn.LogSoftMax())
    fcc = fcc:cuda()
    fcc:evaluate()
    for i =1,Nv,batch do
        xlua.progress(i/batch, Nv/batch)
        local j = math.min(i+batch-1,Nv)
        local Xb = Xv[{{i,j}}]:cuda()
        local Yb = Yv[{{i,j}}]:cuda()
        local out = net:forward(Xb) --out is the deconvoluted image
        --out = out:cuda()
        local untouched = dnet:forward(out)
        untouched = untouched:double()
	    image.display(untouched)
        --analytics:
        --print(out)
        --print(#out[1])
        --y = Ax + b --y und bias spielen aber keine direkte rolle
        --x' = x*((A<max y>)^T/Sum<dim1>(A)^T
        local y = fcc:forward(out)
        local ymax,pos = torch.max(y,2)

        local lmax = i+batch-1 <= Nv and batch or (Nv%batch) --lazy evaluation
        --local tmp = torch.Tensor(128,10,39,39):cuda()
	    local out2 = torch.Tensor(out:size()):copy(out)
	    out2 = out2:cuda()
	    for l=1, lmax do
	    --tmp[l] = out[l]
	    --out[l] = torch.Tensor(10,37,37):fill(0):cuda() 
		--print(out:size())
            out[l][pos[l][1]] = torch.Tensor(25,25):fill(0):cuda() --tmp[l][pos[l][1]]
        end
        
	
	    --local tmp2 = torch.Tensor(128,10,39,39):cuda()
	    for l=1, lmax do
            --tmp2[l] = out2[l]
            --out2[l] = torch.Tensor(10,37,37):fill(torch.max(out2)):cuda()
            out2[l][pos[l][1]] = torch.Tensor(25,25):fill(torch.max(out2)):cuda() --tmp2[l][pos[l][1]]
        end
	    local hnet = dnet:clone()
	

        --dxx:zero()
        --dnet:updateOutput(out2)
        --dnet:clearState()

        local himg = hnet:forward(out2)
        --dxx = dxx:zero()

        local dimg = dnet:forward(out)

        dimg = dimg:double()
        himg = himg:double()

        local sub = torch.Tensor(128,1,imgsize,imgsize)
        local cdimg = torch.Tensor(3,128,imgsize,imgsize)
        local chimg = torch.Tensor(3,128,imgsize,imgsize)
        local cuntouched = torch.Tensor(3,128,imgsize,imgsize)

        for l=1, lmax do
            cdimg[1][l] = dimg[l][1]
            cdimg[2][l] = dimg[l][2]
            cdimg[3][l] = dimg[l][3]
            chimg[1][l] = himg[l][1]
            chimg[2][l] = himg[l][2]
            chimg[3][l] = himg[l][3]
            cuntouched[1][l] = untouched[l][1]
            cuntouched[2][l] = untouched[l][2]
            cuntouched[3][l] = untouched[l][3]

        end

        for l=1, lmax do
            dimg[l][1] = dimg[l][1]:mul(chimg[1]:max()/cdimg[1]:max())
            dimg[l][2] = dimg[l][2]:mul(chimg[2]:max()/cdimg[2]:max())
            dimg[l][3] = dimg[l][3]:mul(chimg[3]:max()/cdimg[3]:max())
            sub[l] = torch.sqrt((himg[l][1]-dimg[l][1]):cmul(himg[l][1]-dimg[l][1])+(himg[l][2]-dimg[l][2]):cmul(himg[l][2]-dimg[l][2])+(himg[l][3]-dimg[l][3]):cmul(himg[l][3]-dimg[l][3]))
            table.insert(list,""..(l+i-1)..","..classifier[Yb[l]]..","..classifier[pos[l][1]])
            --sub[l] = dimg[l][1]:cdiv(untouched[l][1])+dimg[l][2]:cdiv(untouched[l][2])+dimg[l][3]:cdiv(untouched[l][3])
        end
            --print(sub)
        image.display{image=dimg,legend="low"}

	--[=[

        local lmax = i+batch-1 <= Nv and batch or (Nv%batch) --lazy evaluation
        local xD
        local bool
        out = out:double()
        A = A:double()
        pos = pos:long()
        --pos = pos:fill(8)
        --print(pos)
        --print(B)
        --print(pos[1])
        --print((ymax[1]+B[pos[1][1]])[1])
        xD = out[1]
        --xD = torch.cmul(out[1]:div((y[1][8]-B[8])[1]),torch.cdiv(A:index(1,8)[1],torch.sum(torch.abs(A), 1)[1]))
        --nur gt zurückgegen
        --print(torch.min(out[1]))
        --bool = torch.gt(A:index(1,pos[1])[1],torch.sum(A, 1)[1])
        --xD = torch.cmul(out[1],2*bool:double()-1)
        --print(out[1][{{1,30}}])
        --print(xD[{{1,30}}])
        --print(torch.cdiv(A:index(1,pos[1])[1],torch.sum(torch.abs(A), 1)[1])[{{1,30}}])
        --print(A:index(1,pos[1])[1][{{1,30}}])
        --print(torch.sum(torch.abs(A), 1)[1][{{1,30}}])
        -- print(ymax[1],B[pos[1][1]])
        --xD = torch.cmul(out[1],bool:double())
        table.insert(list,""..i..","..classifier[Yv[1+i-1]]..","..classifier[pos[1][1]])
        --local xDc = xD:resize(1,xD:size()[1])
        for l=2, lmax do
            --todo: parellelisieren
            --xD = torch.cmul(out[1],torch.cdiv(A:index(1,pos[1])[1],torch.sum(torch.abs(A), 1)[1]))
            --xD = out[l]
            --xD = torch.cmul(out[l]:div((y[l][8]-B[8])[1]),torch.cdiv(A:index(1,8)[1],torch.sum(torch.abs(A), 1)[1]))
            --bool = torch.gt(A:index(1,pos[l])[1],torch.sum(A, 1)[1])
            --xD = torch.cmul(out[l],2*bool:double()-1)
            --xD = torch.cmul(out[l],bool:double())
            table.insert(list,""..(l+i-1)..","..classifier[Yv[l+i-1]]..","..classifier[pos[l][1]])
            --xDc = torch.cat(xDc, xD:resize(1,xD:size()[1]), 1)
        end

        local xDc = torch.Tensor(128,1,imgsize,imgsize)
        local xD = torch.Tensor(128,10,256*imgsize*imgsize)
        local impulse = torch.Tensor(128,imgsize,imgsize,10,256)
        local value = torch.Tensor(128,imgsize,imgsize)
        local imax = torch.Tensor(128,imgsize,imgsize)
        local ct = torch.Tensor(1):long()
        for b=1,lmax do
            --xlua.progress(b, lmax)
            for c=1,10 do
                ct[1] = c
                xD[b][c] = torch.cmul(out[b],torch.cdiv(A:index(1,ct)[1],torch.sum(torch.abs(A), 1)[1]))
            end
        end
        xD:resize(128,10,256,imgsize,imgsize)
        --xD = xD:cuda()
        --impulse = impulse:cuda()
        --xDc = xDc:cuda()
        for b=1,lmax do
            xlua.progress(b, lmax)
            for c=1,10 do
                for l=1,256 do
                    for iy=1,imgsize do
                        for ix=1,imgsize do
                            impulse = xD:permute(1,4,5,2,3)
                            --impulse[b][iy][ix][c][l] = xD[b][c][l][iy][ix]
                        end
                    end
                end
            end
        end
        for b=1,lmax do
            xlua.progress(b, lmax)
            for iy=1,imgsize do
                for ix=1,imgsize do
                    value[b][iy][ix],imax[b][iy][ix] = torch.max(torch.sum(impulse[b][iy][ix],2),1)
                    xDc[b][1][iy][ix] = imax[b][iy][ix] == pos[b] and value[b][iy][ix] or 0
                end
            end
        end
        --]=]
        --end analytics
        --xDc = xDc:cuda()
        --print(xDc)
        --local img = dnet:forward(xDc)
        --img = img:double()
        --local new = testnet(xDc)
        --new = new:double()
        --local nout = new[1]:fill(0)
        --for f=1,128 do
        --    for g=1,256 do
        --        nout[f] = new[f][g]+nout[f]
        --    end
        --end
        ----no anaylsis test:
        --local img = dnet:forward(out)
        --img = img:double()
        ----end
        ----test
        --local core = testnet:forward(xDc)
        --core = core:double()
        --local corestack = torch.Tensor(3,4,32)
        --local coreline = core[1]
        --local vcmax = 16
        --if(i>Nv-batch) then
        --    vcmax = 2
        --end
        --for vc=1, vcmax do
        --    for wc=2, 8 do
        --        coreline = torch.cat(coreline,img[vc*8-8+wc],3)
        --    end
        --    corestack = torch.cat(corestack,coreline,2)
        --    coreline = core[vc*8-8+1]
        --end
        --print(imgstack)
        --image.save("coreimg"..i..".png",corestack)
        ----testend

        --print(img[1])
        --image.save("img "..i,img[1])
        --local write = image.compressJPG(img[1])
        --file = torch.DiskFile('img'..i..'.asc', 'w')
        --file:writeObject(img)
        --file:close() -- make sure the data is writte
        --local imgstack = torch.Tensor(3,imgsize,imgsize*8)
        --local imgline = img[1]
        --local vmax = 16
        --if(i>Nv-batch) then
        --    vmax = 2
        --end
        --for v=1, vmax do
        --    for w=2, 8 do
        --        imgline = torch.cat(imgline,img[v*8-8+w],3)
        --    end
        --    imgstack = torch.cat(imgstack,imgline,2)
        --    imgline = img[v*8-8+1]
        --end
        --print(imgstack)
        --imgstack = imgstack/torch.mean(imgstack)
        --image.save("img"..i..".png",imgstack)

        --image.display(img)
        --for o=1,128 do
        --  for f=1,3 do
        --    for g=1,32 do
        --      for h=1,32 do
        --        if img[o][f][g][h] < 0 then
        --         img[o][f][g][h] = 0
        --        end
        --      end
        --    end
        --  end
        --end
        --print(img:size())
        --print(img[1][1][1][{{10,30}}])
        image.display{image=himg,legend="high"}
        --img = img:cuda()
        --local bw = testnet:forward(img)
        --image.display(bw)
        --image.display(untouched-img)
	    image.display{image=sub,legend="sub"}
        --image.display{image=dimg,legend="low"}
        image.display{image=Xb,legend="original"}
	    image.display{image=(himg-dimg),legend="dsub"}
    end
    for i=1,#list do
        print(list[i])
    end
end

return decode
