% We update the network parameters in this version of OSBDnet with manually derived gradients
% The corresponding code can be replaced with dlfeval+dlgradient in MATLAB versions that support automatic derivation
function [Output, w_out, time, loss_mse] = OSBDnet(x_rx, para)
% Take out the parameters from the structure
Itr = para.Itr;
filter_len = para.filter_len;
U = para.U;
init_lr = para.init_lr;
R_max = para.R_max;
I_max = para.I_max;
warmstart_switch = para.warmstart_switch;
warmstart_lr = para.warmstart_lr;
warmstart_epoch = para.warmstart_epoch;
paddingMode = para.paddingMode;
Std = para.std;
Sign_rate = para.Sign_rate;
sign_length = para.sign_length;
stop_norm = para.stop_norm;
padding = ( filter_len - 1 ) / 2 ;
if isfield(para, 'MSE_loss')
    ifMSE = para.MSE_loss;
else
    ifMSE = true;
end
% Parameters initialization
beta1 = 0.9;  % Adam Parameters
beta2 = 0.99;
epsilon = 1e-8;
avg_g = 0;
avg_gsq =0;
avg_g2 = 0;
avg_gsq2 =0;
w_init = [zeros(padding, 1); 1; zeros(padding, 1)] ;  % Initialize Filter Vector
pretraining_lr_list = [warmstart_lr: (init_lr - warmstart_lr)/warmstart_epoch: init_lr];
Para = zeros(filter_len + 2*U+1 + 3, 1);  % OSBDnet's learnable parameters    format：[w; Chi; beta; alpha; eta]
Chi = 1./(2*[-U:U]+[zeros(1,U),1e15,zeros(1,U)]).^2; % Chi
Para(filter_len+1: filter_len+2*U+1) = -log((1 ./ (Chi + 1e-8)) - 1);  % filter
Para(filter_len+2*U+2) = 20;  % beta  defalt: 20
Para(filter_len+2*U+3) = 5;  % alpha
Para(filter_len+2*U+4) = 5;  % eta
gsq_max = zeros(size(Para));
% Zeroing for equalized convolution
if strcmp(paddingMode, 'noncasual')
    Input = [zeros(padding, 1); x_rx; zeros(padding, 1)] ;
else
    Input = [zeros(padding*2, 1); x_rx; ] ;
end
w_old = 10;
Y = zeros(1, sign_length*Sign_rate);  % Received Signal Vector
X = zeros(filter_len, sign_length*Sign_rate);  % Equalized Signal Vector
flag_eta = 0;
flag_pt = warmstart_switch;
lr = init_lr;
tic
% Constructing the convolution matrix
for j = 1 : sign_length * Sign_rate
    X(:, j) = Input(j: j+filter_len-1);  % X: (filter_len, sign_length*Sign_rate)
end
N = size(X, 2);

for i = 1 : Itr
    % Learning rate adjustment
    if flag_pt && i <= warmstart_epoch
        lr = pretraining_lr_list(i);
    elseif flag_pt 
        flag_pt = false;
        lr = init_lr;
    end
    if i>3 && abs(sigma_list(i-1)-sigma_list(i-2))<0.001 && flag_eta > 1 && flag_pt==false
        flag_eta = 0;
        lr = lr*0.8;
    end
    flag_eta = flag_eta + 1;
    % ============ forward propagation and back propagation==============
    % Parameters：
    delta_W = Para(1: filter_len);
    alpha = Sigmoid(Para(filter_len+2*U+3));
    beta = Sigmoid(Para(filter_len+2*U+2));
    eta = Sigmoid(Para(filter_len+2*U+4));
    Chi = Sigmoid(Para(filter_len+1: filter_len+2*U+1))+[zeros(U,1);1;zeros(U,1)];
    % Updating filters
    W_new = (alpha)*delta_W;
    W_r = real(W_new);
    W_i = imag(W_new);
    % Perform convolution for linear equalization
    X_r = real(X);
    X_i = imag(X);
    x_ori = X((filter_len-1)/2+1, :);
    y_r_ = (W_r.'*X_r - W_i.'*X_i);
    y_i_ = (W_r.'*X_i + W_i.'*X_r);
    y_r = y_r_ + eta*real(x_ori) ;
    y_i = y_i_ + eta*imag(x_ori);
    O = y_r + 1j*y_i;
    O = O / std(O) * Std;
    % Find the nearest constellation point
    Range = [-2*U:2:2*U].';
    d_i_r = floor(real(O));
    d_i_r = d_i_r+1-mod(d_i_r, 2);
    d_i_r(d_i_r>R_max) = R_max;
    d_i_r(d_i_r<-R_max) = -R_max;
    d_i_i = floor(imag(O));
    d_i_i = d_i_i+1-mod(d_i_i, 2);
    d_i_i(d_i_i>I_max) = I_max;
    d_i_i(d_i_i<-I_max) = -I_max;
    d_i = d_i_r + 1j*d_i_i;
    d_ir_mtx = d_i_r + Range;
    d_ir_mtx(abs(d_ir_mtx)>R_max)=0;
    d_ii_mtx = d_i_i + Range;
    d_ii_mtx(abs(d_ii_mtx)>I_max)=0;
    %Calculate the loss function
    A_r = y_r - d_ir_mtx;
    A_i = y_i - d_ii_mtx;
    D_r = d_ir_mtx.^2;
    D_i = d_ii_mtx.^2;
    AD_r = sum(Chi.*A_r.*D_r, 1);
    AD_i = sum(Chi.*A_i.*D_i, 1);
    res_r = y_r-d_i_r;
    res_i = y_i-d_i_i;
    V_r = Chi.' * (abs(A_r).^2 .* D_r);
    V_i = Chi.' * (abs(A_i).^2 .* D_i);
    e_R = sum(V_r, 'all');  % real part of the loss function
    e_I = sum(V_i, 'all');  % imaginary part of the loss function
    MSE = sum(abs(y_r - real(Y)).^2) + sum(abs(y_i - imag(Y)).^2);   % MSE loss
    loss_sbd = sqrt((e_R + e_I)/N);  
    loss_mse =  sqrt(MSE / N);
    % gradient calculation
    if ~ifMSE  % If the MSE loss function is not used, the corresponding weighting factor is set to zero
        Para(filter_len+2*U+2)=20;
        beta=1;
    end
    Grad_dWr_S = (1 / loss_sbd) * (alpha*(X_r*AD_r'+X_i*AD_i'));
    Grad_dWi_S = (1 / loss_sbd) * (alpha*(X_r*AD_i'-X_i*AD_r'));
    Grad_dWr_M = (1 / loss_mse) * alpha * (X_r*res_r'+X_i*res_i');
    Grad_dWi_M = (1 / loss_mse) * alpha * (X_r*res_i' - X_i*res_r');
    Grad_dWr = beta*Grad_dWr_S + (1-beta)*Grad_dWr_M+0.9^i*randn(size(Grad_dWr_S));
    Grad_dWi = beta*Grad_dWi_S + (1-beta)*Grad_dWi_M+0.9^i*randn(size(Grad_dWr_S));
    Grad_chi = (0.5 / loss_sbd) * sum(A_r.^2.*D_r + A_i.^2.*D_i, 2).*Chi.*(1-Chi);
    Grad_beta = (loss_sbd-loss_mse)*beta*(1-beta)*N;
    Grad_alpha = (1 / loss_sbd) *(1-alpha) * (AD_r*y_r_'+AD_i*y_i_') * beta +...
        (1 / loss_mse) * (res_r*y_r_'+res_i*y_i_')* (1-beta) *(1-alpha);
    Grad_eta = (1 / loss_sbd) *eta*(1-eta) * (AD_r*real(x_ori)'+AD_i*imag(x_ori)') * beta +...
        (1 / loss_mse) * (res_r*real(x_ori)'+res_i*imag(x_ori)')* (1-beta) *eta*(1-eta);
    % Adam parameter Updating 
    % note that : Para = [w; Chi; beta; alpha; eta]
    Grad_Para = [Grad_dWr+ 1j*Grad_dWi; Grad_chi; Grad_beta; Grad_alpha; Grad_eta]/N;
    avg_g = beta1.*avg_g + (1 - beta1).*Grad_Para;  % Updating the first-order estimate of the bias
    avg_gsq= beta2.*avg_gsq + (1 - beta2).*(abs(Grad_Para).^2);  % Updating second-order moment estimates for bias
    biasCorrection = sqrt(1-beta2.^(i))./(1-beta1.^(i));  
    effectiveLearnRate = biasCorrection.*lr;
    step = -effectiveLearnRate.*( avg_g./(sqrt(avg_gsq+ epsilon) ) );
    Para = Para+step;
    w_ = Para(1: filter_len);
    delta_w = sum(abs(w_-w_old).^2, 'all');
    w_old = w_;
    % Updating the equalization results
    Y = d_i;
    % early termination
    if (lr < 1e-3 || delta_w<stop_norm ) && flag_pt == false  % 1e-3
        break
    end
end
w_out = alpha*w_ + eta*w_init;
Output = O.';
Output = reshape(Output.', Sign_rate, sign_length).' ;
Output = sum(Output, 2);
time = toc;
ItrNum = i;
end

%% Sigmoid Function
function y = Sigmoid(x)
    y=1./(1+exp(-x));
end