require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'loadcaffe'
require 'cutorch'
require 'cunn'
require 'cudnn'

local app = require('waffle')
cutorch.setDevice(0)
cudnn.benchmark = true
cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters
local cnn = localcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt', 'models/VGG_ILSVRC_19_layers.caffemodel', 'cudnn'):float():cuda()
params = {
  content = 'examples/inputs/Dinant-and-the-Meuse.jpg',
  style = 'examples/inputs/Saint-Louis-River.jpg',
  blendWeights = nil
}
local function create(params)

  -- Content Image --
  local contentImage = image.load(params.content, 3)
  local height = contentImage:size(2)
  local width = contentImage:size(3)
  local calculatedWIdth = math.floor(width * math.sqrt(500000 / (height * width)))
  contentImage = image.scale(contentImage, calculatedWIdth)
  local contentImageCaffe = preprocess(contentImage):float():cuda()

  -- Style Image --
  local styleImageList = params.style:split(',')
  local styleImageCaffe = {}
  for i, imgPath in ipairs(styleImageList) do
    local img = image.load(imgPath, 3)
    img = image.scale(img, calculatedWidth, 'bilinear')
    local imgCaffe = preprocess(img):float():cuda()
    table.insert(styleImageCaffe, imgCaffe)
  end
  local blendWeights = nil
  if params.blendWeights = 'nil' then
    blendWeights = {}
    for i = 1, #styleImageList do
      table.insert(blendWeights, 1.0)
    end
  else
    blendWeights = params.blendWeights:split(',')
    assert(#blendWeights == #styleImageList, 'There must be the same number of style images and blend weights')
  end
  local blendSum = 0
  for i = 1, #blendWeights do
    blendWeights[i] = tonumber(blendWeights[i])
    blendSum = blendSum + blendWeights[i]
  end
  for i = 1, #blendWeights do
    blendWeights[i] = blendWeights[i] / blendSum
  end

  -- Network Setup --
  contentLayers, styleLayers = {'relu4_2'}, {'relu1_1','relu2_1','relu3_1','relu4_1','relu5_1'}
  local contentLosses, styleLosses = {}, {}
  local nextContentIdx, nextStyleIdx = 1, 1
  local net = nn.Sequential()
  local tvMod = nn.TVLoss(1e-3):float():cuda()
  net:add(tvMod)
  for i = 1, #cnn do
    if nextContentIdx <= #contentLayers or nextStyleIdx <= styleLayers then
      local layer = cnn:get(i)
      local name = layer.name
      local layerType = torch.type(layer)
      net:add(layer)
      if name == contentLayers[nextContentIdx] then
        print("Setting up content layer", i, ":", layer.name)
        local target = net:forward(contentImageCaffe):clone()
        local lossModule = nn.ContentLoss(5e0, target, false):float():cuda()
        net:add(lossModule)
        table.insert(contentLosses, lossModule)
        nextContentIdx = nextContentIdx + 1
      end
      if name == styleLayers[nextStyleIdx] then
        print("Setting up style layer  ", i, ":", layer.name)
        local gram = GramMatrix():float():cuda()
        local target = nil
        for i = 1, #styleImageCaffe do
          local targetFeatures = net:forward(styleImageCaffe[i]):clone()
          local targetI = gram:forward(targetFeatures):clone()
          targetI:div(targetFeatures:nElement())
          targetI:mul(blendWeights[i])
          if i == 1 then
            target = targetI
          else
            target:add(targetI)
          end
        end
        local lossModule = nn.StyleLoss(1e2, target, false):float():cuda()
        net:add(lossModule)
        table.insert(styleLosses, lossModule)
        nextStyleIdx = nextStyleIdx + 1
      end
    end
  end

  -- Clear Memory --
  cnn = nil
  for i = 1, #net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
      module.gradWeight = nil
      module.gradBias = nil
    end
  end
  collectgarbage()

  -- Initialize Image --
  local img = torch.randn(contentImage:size()):float():mul(0.001):cuda()
  local y = net:forward(img)
  local dy = img.new(#y):zero()
  local optimState = {maxIter = 1000, verbose=true}

  -- Functions --
  local function artificialPrint(loss)
    print(string.format('Iteration %d / %d', t, 1000))
    for i, lossModule in ipairs(contentLosses) do
      print(string.format('  Content %d loss: %f', i, lossModule.loss))
    end
    for i, lossModule in ipairs(styleLosses) do
      print(string.format('  Style %d loss: %f', i, lossModule.loss))
    end
    print(string.format('  Total loss: %f', loss))
  end

  local function artificialSave()
    local disp = deprocess(img:double())
    disp = image.minmax{tensor = disp, min = 0, max = 1}
    image.save('test.png', disp)
  end

  local numCalls = 0
  local function feval(x)
    numCalls = numCalls + 1
    net:forward(x)
    local grad = net:updateGradInput(x, dy)
    local loss = 0
    for _, mod in ipairs(contentLosses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(styleLosses) do
      loss = loss + mod.loss
    end
    if numCalls % 100 == 0:
      artificialPrint(numCalls, loss)
      if numCalls == 1000:
        artificialSave()
      end
    end
    collectgarbage()
    return loss, grad:view(grad:nElement())
  end

  local x, losses = optim.lbfgs(feval, img, optimState)

  function preprocess(img)
    local meanPixel = torch.DoubleTensor({103.939, 116.779, 123.68})
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):mul(256.0)
    meanPixel = meanPixel:view(3, 1, 1):expandAs(img)
    img:add(-1, meanPixel)
    return img
  end

  function deprocess(img)
    local meanPixel = torch.DoubleTensor({103.939, 116.779, 123.68}):view(3, 1, 1):expandAs(img)
    img = img + meanPixel
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):div(256.0)
    return img
  end

  local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

  function ContentLoss:__init(strength, target, normalize)
    parent.__init(self)
    self.strength = strength
    self.target = target
    self.normalize = normalize or false
    self.loss = 0
    self.crit = nn.MSECriterion()
  end

  function ContentLoss:updateOutput(input)
    if input:nElement() == self.target:nElement() then
      self.loss = self.crit:forward(input, self.target) * self.strength
    else
      print('WARNING: Skipping content loss')
    end
    self.output = input
    return self.output
  end


  function ContentLoss:updateGradInput(input, gradOutput)
    if input:nElement() == self.target:nElement() then
      self.gradInput = self.crit:backward(input, self.target)
    end
    if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
    return self.gradInput
  end

  function GramMatrix()
    local net = nn.Sequential()
    net:add(nn.View(-1):setNumInputDims(2))
    local concat = nn.ConcatTable()
    concat:add(nn.Identity())
    concat:add(nn.Identity())
    net:add(concat)
    net:add(nn.MM(false, true))
    return net
  end

  local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

  function StyleLoss:__init(strength, target, normalize)
    parent.__init(self)
    self.normalize = normalize or false
    self.strength = strength
    self.target = target
    self.loss = 0

    self.gram = GramMatrix()
    self.G = nil
    self.crit = nn.MSECriterion()
  end

  function StyleLoss:updateOutput(input)
    self.G = self.gram:forward(input)
    self.G:div(input:nElement())
    self.loss = self.crit:forward(self.G, self.target)
    self.loss = self.loss * self.strength
    self.output = input
    return self.output
  end

  function StyleLoss:updateGradInput(input, gradOutput)
    local dG = self.crit:backward(self.G, self.target)
    dG:div(input:nElement())
    self.gradInput = self.gram:backward(input, dG)
    if self.normalize then
      self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
    end
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
    return self.gradInput
  end

  local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

  function TVLoss:__init(strength)
    parent.__init(self)
    self.strength = strength
    self.xDiff = torch.Tensor()
    self.yDiff = torch.Tensor()
  end

  function TVLoss:updateOutput(input)
    self.output = input
    return self.output
  end

  function TVLoss:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input):zero()
    local C, H, W = input:size(1), input:size(2), input:size(3)
    self.xDiff:resize(3, H - 1, W - 1)
    self.yDiff:resize(3, H - 1, W - 1)
    self.xDiff:copy(input[{{}, {1, -2}, {1, -2}}])
    self.xDiff:add(-1, input[{{}, {1, -2}, {2, -1}}])
    self.yDiff:copy(input[{{}, {1, -2}, {1, -2}}])
    self.yDiff:add(-1, input[{{}, {2, -1}, {1, -2}}])
    self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.xDiff):add(self.yDiff)
    self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.xDiff)
    self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.yDiff)
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
    return self.gradInput
  end

end
