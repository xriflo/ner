require 'torch'
require 'nn'
require 'optim'
require 'os'
npy4th = require 'npy4th'

function shuffle_dataset(dataset, targets)
	local n = dataset:size()[1]
	shuffled_ind = torch.randperm(n)
	shuffled_dataset = torch.Tensor(dataset:size())
	shuffled_targets = torch.Tensor(targets:size())
	for i=1,n do
		shuffled_dataset[i] = dataset[shuffled_ind[i]]
		shuffled_targets[i] = targets[shuffled_ind[i]]
	end
	return shuffled_dataset, shuffled_targets
end

function join_datasets(datasets_per_class, targets_per_class, how, ninputs)
	if how=="balanced" then
		min = math.huge
		for i=1,#datasets_per_class do
			if datasets_per_class[i]:size()[1]<min then
				min = datasets_per_class[i]:size()[1]
			end
		end
		samples_per_train = math.floor(0.8*min)
		samples_per_test = min-samples_per_train
		dataset_train = torch.Tensor(samples_per_train*#datasets_per_class, ninputs)
		targets_train = torch.Tensor(samples_per_train*#datasets_per_class)
		dataset_test = torch.Tensor(samples_per_test*#datasets_per_class, ninputs)
		targets_test = torch.Tensor(samples_per_test*#datasets_per_class)

		for i=1,#datasets_per_class do
			for j=1,samples_per_train do
				dataset_train[(i-1)*samples_per_train+j] = datasets_per_class[i][j]
				targets_train[(i-1)*samples_per_train+j] = targets_per_class[i][j]
			end

			for j=1, samples_per_test do
				dataset_test[(i-1)*samples_per_test+j] = datasets_per_class[i][j+samples_per_train]
				targets_test[(i-1)*samples_per_test+j] = targets_per_class[i][j+samples_per_train]
			end
		end
		dataset_train, targets_train = shuffle_dataset(dataset_train, targets_train)
		dataset_test, targets_test = shuffle_dataset(dataset_test, targets_test)

		trainData = {
	   		data = dataset_train,
	   		labels = targets_train,
	   		size = function() return samples_per_train*#datasets_per_class end
		}

		testData = {
			data = dataset_test,
			labels = targets_test,
			size = function() return samples_per_test*#datasets_per_class end
		}

		trainData.data = trainData.data:double()
		testData.data = testData.data:double()
		print (trainData.data:size())
		print (trainData.labels:size())
		print (testData.data:size())
		print (testData.labels:size())
		return trainData, testData
	end
	return nil, nil
end

function process_dataset(dataset, targets, how)
	nclasses = 5
	nsamples = dataset:size()[1]
	ninputs = dataset:size()[2]

	-- compute samples per class
	no_samples_per_class = {[1]=0, [2]=0, [3]=0, [4]=0, [5]=0}
	no_examples_per_train = {}
	no_examples_per_test = {}

	for i=1,nsamples do
		no_samples_per_class[targets[i]] = no_samples_per_class[targets[i]]+1
	end

	for i=1,#no_samples_per_class do
		no_examples_per_train[i] = math.floor(0.8*no_samples_per_class[i])
		no_examples_per_test[i] = no_samples_per_class[i]-no_examples_per_train[i]
	end

	-- gather samples per class
	datasets_per_class = {}
	targets_per_class = {}

	for i=1,nclasses do
		datasets_per_class[i] = torch.Tensor(no_samples_per_class[i],ninputs)
		targets_per_class[i] = torch.Tensor(no_samples_per_class[i], 1)
	end

	for i=1,nclasses do
		numpy_index = 1
		for j=1,nsamples do
			if targets[j]==i then
				datasets_per_class[i][numpy_index] = dataset[j]
				targets_per_class[i][numpy_index] = targets[j]
				numpy_index = numpy_index + 1
			end
		end
	end

	-- shuffle dataset per classes
	for i=1,nclasses do
		datasets_per_class[i], targets_per_class[i] = shuffle_dataset(datasets_per_class[i], targets_per_class[i])
	end

	trainData, testData = join_datasets(datasets_per_class, targets_per_class, how, ninputs)
	return trainData, testData
end

dataset = npy4th.loadnpy('dataset.in.npy')
targets = npy4th.loadnpy('targets.in.npy')
targets = targets + 1

length_vec = 80
nhiddens = 800
noutputs = 5

classes = {'1','2','3','4','5'}
confusion = optim.ConfusionMatrix(classes)

trainData, testData = process_dataset(dataset, targets, "balanced")
trsize = trainData:size()
tesize = testData:size()

model = nn.Sequential()
model:add(nn.Linear(ninputs,ninputs*8))
model:add(nn.ReLU())
model:add(nn.Linear(ninputs*8,ninputs*4))
model:add(nn.ReLU())
model:add(nn.Linear(ninputs*4,noutputs))

print(model)
local method = 1
criterion = nn.MultiMarginCriterion()
parameters,gradParameters = model:getParameters()

optimState = {
	learningRate = 0.99,
	weightDecay = 0,
	momentum = 0.1,
	learningRateDecay = 1e-4
}
optimMethod = optim.sgd
batchSize = 10

local method = 'xavier'
model = require('weight-init')(model, method)

function train()
   epoch = epoch or 1
   local time = sys.clock()

   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,trainData:size(),batchSize do
      	xlua.progress(t, trainData:size())

		-- create mini batch
		local inputs = {}
		local targets = {}
		for i = t,math.min(t+batchSize-1,trainData:size()) do
			-- load new sample
			local input = trainData.data[shuffle[i]]
			local target = trainData.labels[shuffle[i]]
				table.insert(inputs, input)
				table.insert(targets, target)
		end

		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
			-- get new parameters
			if x ~= parameters then
			  	--parameters:copy(x)
			end

			-- reset gradients
			gradParameters:zero()

		   	-- f is the average of all criterions
			local f = 0

			-- evaluate function for complete mini batch
			for i = 1,#inputs do
			  -- estimate f
			  local output = model:forward(inputs[i])
			  local err = criterion:forward(output, targets[i])
			  f = f + err

			  -- estimate df/dW
			  local df_do = criterion:backward(output, targets[i])
			  model:backward(inputs[i], df_do)

			  -- update confusion
			  confusion:add(output, targets[i])
			end

			-- normalize gradients and f(X)
			gradParameters:div(#inputs)
			f = f/#inputs

		    -- return f and df/dX
		    return f,gradParameters
		end

      	-- optimize on current mini-batch
        optimMethod(feval, parameters, optimState)
   	end

	-- time taken
	time = sys.clock() - time
	time = time / trainData:size()

	-- print confusion matrix
	print(confusion)
	print("accuracy=" .. confusion.totalValid * 100)

	-- save/log current net
	local filename = paths.concat('results', 'model.net')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	print('==> saving model to '..filename)
	torch.save(filename, model)

	-- next epoch
	confusion:zero()
	epoch = epoch + 1
end

function test()
	local time = sys.clock()
	for t = 1,testData:size() do
		local input = testData.data[t]
		local target = testData.labels[t]
		local pred = model:forward(input)
      	confusion:add(pred, target)
      	time = sys.clock() - time
	    time = time / testData:size()
	end
	print(confusion)
	print("accuracy=" .. confusion.totalValid * 100)
	confusion:zero()
end



for i=1,300 do
	train()
	test()
end