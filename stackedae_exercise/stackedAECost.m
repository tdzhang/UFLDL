function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%



forwardA = cell(1,numel(stack));
forwardA{1} = data;
det = cell(1,numel(stack)+1);
z = stack{1}.w*data + repmat(stack{1}.b,1,M);
for d = 2:numel(stack)
    a = sigmoid(z);
    z = stack{d}.w*a + repmat(stack{d}.b,1,M);
    forwardA{d} = a;  
end
a = sigmoid(z);

h = softmaxTheta*a;
h = bsxfun(@minus, h, max(h, [], 1));
h = exp(h);
h = bsxfun(@rdivide, h, sum(h));

softmaxThetaGrad = (-a*(groundTruth'-h')/M + lambda*softmaxTheta')';


det{numel(stack)+1} = -softmaxTheta'*(groundTruth - h).*a.*(1-a); 
for d = numel(stack):-1:2
    det{d} = stack{d}.w'*det{d+1}.*forwardA{d}.*(1-forwardA{d});
end

for d = 1:numel(stack)
    stackgrad{d}.w = det{d+1}*forwardA{d}'/M;
    stackgrad{d}.b = sum(det{d+1},2)/M;
end
cost = -sum(sum(groundTruth.*log(h)))/M + 0.5*lambda*sum(sum(softmaxTheta.^2));














% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
