clear;
data = load("demand-2024_8.txt");
%% Question 1
% Level model
year1_data = data(1:52);
year2_data = data(53:end);

year1_length = round(0.8 * length(year1_data));
year1_train = year1_data(1:year1_length, :);
year1_test = year1_data(year1_length+1:end, :);

% Divide year2_data into training and testing sets
year2_length = round(0.8 * length(year2_data));
year2_train = year2_data(1:year2_length, :);
year2_test = year2_data(year2_length+1:end, :);


% Corrected indexing for test data
y_1_hat_year1 = mean(year1_data(1:10));
y_1_hat_year2 = mean(year2_data(1:10));
[alpha_1, mse_1] = find_optimal_alpha(year1_train, y_1_hat_year1);
[alpha_2, mse_2] = find_optimal_alpha(year2_train, y_1_hat_year2);

year1_predictions = simple_exponential_smoothing(year1_test, alpha_1);
year2_predictions = simple_exponential_smoothing(year2_test, alpha_2);
mse_value1 = mean((year1_test' - year1_predictions).^2);
mse_value2 = mean((year2_test' - year2_predictions).^2);

fprintf('The RMSE of Level SES model for year 2 is: %d\n', sqrt(mse_value2));

smoothed_data = simple_exponential_smoothing(year2_data, alpha_2);

pred = alpha_2 * year2_data(end) + (1-alpha_2) * smoothed_data(end);
Y = zeros(1,6);
for j = 1:6
    Y(j) = pred * j;
end

naive_error = zeros(1,6);
for k = 1:6
    naive_error(k) = sqrt(mse_2) * k;
end
%% Trend model

[mse_year1, alpha1, beta1, results1] = reg(year1_train);
[mse_year2, alpha2, beta2, results2] = reg(year2_train);

year1_trendpred = predict(year1_test, results1, alpha1, beta1);
year2_trendpred = predict(year1_test, results2, alpha2, beta2);

mse_value1_trend = mean((year1_test - year1_trendpred).^2);
mse_value2_trend = mean((year2_test - year2_trendpred).^2);

fprintf('The RMSE of trend model for year 2 is: %d\n', sqrt(mse_value2_trend));

% Predictions
l = results2(1);
b = results2(2);
h = 1:6;
predictions = l+h*b;

agg_predictions = cumsum(predictions);

%% Question 2 
S = 32000;
v = 120;
profit_margin = 1.1;
p = v*profit_margin;
L = 2; % weeks
R = 4; % weeks
r = 0.25; % per year
r_week = 0.25/52; % per week
A = 500; % order costs 500 per order
B2 = 0.05;

mu_R = agg_predictions(4);
sigma_R = naive_error(4);

mu_L = L/R*mu_R;
sigma_L = sqrt(L/R)*sigma_R;

mu_RL    = (1+L/R)*mu_R;
sigma_RL = sqrt(1+L/R)*sigma_R;

k = @(S) (S-mu_RL)/sigma_RL;
k_value = k(S);

rho_R = (mu_R/sigma_R)^2;
rho_L = rho_R*L/R;
rho_RL = (R+L)*rho_R / R;
lambda_R = mu_R / (sigma_R^2);

P1 = @(S) gamcdf(S,rho_RL,1/lambda_R);
P1_value = P1(S);

ESPRC = @(S) loss_gamma(S,rho_RL,lambda_R)-loss_gamma(S,rho_L,lambda_R);
ESPRC_value = ESPRC(S);

P2 = @(S) 1 - (ESPRC(S)/mu_R);
P2_value = P2(S);

order = @(S) A*52/R;
order_value = order(S);

hold = @(S) (k(S)*sigma_RL+(1/2)*mu_R)*v*r;
hold_initial = hold(S);


shortage = @(S) B2*v*ESPRC(S)*(52/R);
shortage_value = shortage(S);

TC = @(S) order(S) + hold(S) + shortage(S);
TC_value=TC(S);

%% Question 3
TC = @(S) A*52/R + (S-mu_RL+0.5*mu_R)*v*r + B2*v*ESPRC(S)*52/R;

S_values = 1:100000; 

values = arrayfun(TC,S_values);
[minimum_TC, index] = min(values);
optimal_S= S_values(index);

P1_opt = P1(optimal_S);
P2_opt = P2(optimal_S);
order_value_opt = order(optimal_S);
hold_value_opt = hold(optimal_S);

ESPRC = @(S) loss_gamma(S,rho_RL,lambda_R)-loss_gamma(S,rho_L,lambda_R);
ESPRC_value_optS = ESPRC(optimal_S);

shortage_value_opt = shortage(optimal_S);
TC_opt = TC(optimal_S);

%% Question 4
P2_value = 0.99;

service_eqn = @(S) ESPRC(S) - (1-P2_value)*mu_R;
S_opt_P2 = fzero(service_eqn, mu_RL);

ESPRC = @(S) loss_gamma(S,rho_RL,lambda_R)-loss_gamma(S,rho_L,lambda_R);
ESPRC_value_optS = ESPRC(S_opt_P2);

P1_opt_P2 = P1(S_opt_P2);
P2_opt_P2 = P2(S_opt_P2);
order_opt_P2 = order(S_opt_P2);
hold_opt_P2 = hold(S_opt_P2);
shortage_opt_P2 = shortage(S_opt_P2);
total_opt_P2 = TC(S_opt_P2);

%% Question 5
B1_from_S = @(S) (v*r*R)/(52*gampdf(S,rho_RL,1/lambda_R)); % Get B1 for any given S value
P2_values = linspace(0.2,0.99,1000); % generate P2 values 0.99 -target value, low values are unreasonable
goodwill = zeros(1, length(P2_values)); % empty one-dimensional vector to store values

for i = 1:length(P2_values) % for all P2 values
    serv = @(S) ESPRC(S) - (1-P2_values(i))*mu_R; % Service equation
    S_implied_P2 = fzero(serv, mu_RL); % We find for which value of S it is zero
    goodwill(i) = B1_from_S(S_implied_P2)/ESPRC(S_implied_P2); % Using the optimal S, we calculate goodwill 
end

B1_from_S_opt=B1_from_S(S_opt_P2);
B1_from_S_ini=B1_from_S(32000);
plot(P2_values,goodwill);
xlabel('Fill Rate (%)');
ylabel('Goodwill Costs per Item Short');

fprintf('The implied goodwill cost at the fill rate target 99%: %d\n', goodwill(end))
fprintf('Initial B1: %d\n:', B1_from_S(32000));
fprintf('Optimal B1: %d\n:', B1_from_S(S_opt_P2));

%% Question 6 
% No shortages, so Total costs = order costs + holding costs
%looking for minimum costs given no shortage costs given review period of 1
%to 52 weeks
order_costs = @(R) A*52./R; %52 weeks
holding_costs = @(R) max(0,(S-Y(1)*(L+R)+0.5*Y(1)).*R)*v*r;
total_costs = @(R) holding_costs(R) + order_costs(R);   
TC_all = total_costs(1:52);

[minimum_TC, minimum_R] = min(TC_all);

mu_R = Y(1)*minimum_R;
sigma_R = naive_error(1)*sqrt(minimum_R);
mu_L = mu_R*L/minimum_R;
sigma_L = sqrt((sigma_R^2)*(L/minimum_R));
mu_RL = mu_R + mu_L;
sigma_RL = sqrt(sigma_R^2 + sigma_L^2);

rho_R = (mu_R/sigma_R)^2;
rho_RL = (mu_RL/sigma_RL)^2;
rho_L = (mu_L/sigma_L)^2;
lambda_R = (mu_R)/(sigma_R^2);

P2 = 0.99;
service_eqn_R = @(S) ESPRC(S)-(1-P2)*mu_R; 
S_new = fzero(service_eqn_R, mu_RL);

P1_new = gamcdf(S_new, rho_RL, 1/lambda_R);
P2_new = 1 - ESPRC(S_new)/mu_R;
ESPRC_6=ESPRC(S_new);
o_costs = A*52/minimum_R;
h_costs = (k(S_new)*sigma_RL+(1/2)*mu_R)*v*r;
s_costs = B2*v*ESPRC(S_new)*52/minimum_R;
goodwill_costs = ESPRC(S_new)*goodwill(end);

TC_question6 = o_costs+h_costs+s_costs+goodwill_costs;


%% Question 8
%s,Q
P1 = @(k) normcdf(k,0,1);
P1_value_sq = P1(k_value);

ESPRC_n = @(S) sigma_RL*(normcdf(k_value,0,1)-k_value*normcdf(-k_value,0,1));
ESPRC_value_sq = ESPRC_n(S);

P2 = @(S) 1 - (ESPRC_n(S)/mu_R);
P2_value_sq = P2(S);

order = @(S) A*mean(data)/S;
order_value_sq = order(S)*52;

hold_initial_sq=0.5*S*v*r;

shortage_SQ = B2*v*ESPRC_n(S)*(52/R);

total_cost_sQ = order_value_sq+hold_initial_sq+shortage_SQ;

%s,S
EOQ=sqrt((mean(data)*120)+500/(mean(data)*120*0.25))*52;
ESPRC_value_ss = ESPRC_n(EOQ);
k_val_SS=k(EOQ);
P1_val_SS=P1(k_val_SS);
P2_val_ss=P2(EOQ);
order_value_ss=order(EOQ)*52;
hold_initial_SS=0.5*EOQ*v*r;
shortage_SS = B2*v*ESPRC_n(EOQ)*(52/R);
total_cost_SS=order_value_ss+hold_initial_SS+shortage_SS;


%% Functions
function smoothed_data = simple_exponential_smoothing(data, alpha)
    % Initialize the smoothed data with the first observation
    smoothed_data = data(1);
    
    % Perform simple exponential smoothing
    for i = 2:length(data)
        smoothed_data(i) = alpha * data(i) + (1 - alpha) * smoothed_data(i-1);
    end
end

function [optimal_alpha, optimal_mse] = find_optimal_alpha(data_train, y_1_hat)
    % Define a range of alpha values (0 to 1)
    alphas = linspace(0, 1, 1000);
    
    % Initialize minimum MSE and corresponding alpha
    min_mse = inf;
    optimal_alpha = 0;
    y_hat = zeros(size(data_train));
    % Arrays to store alpha and corresponding MSE values
    alpha_values = zeros(size(alphas));
    mse_values = zeros(size(alphas));

    % Iterate over each alpha and calculate MSE
    for i = 1:length(alphas)
        alpha = alphas(i);
        % Initialize predictions with y_1_hat
        y_hat(1) = y_1_hat;
        
        % Calculate predictions for the remaining data points
        for t = 2:length(data_train)
            y_hat(t) = alpha * data_train(t-1) + (1 - alpha) * y_hat(t-1);
        end
        
        % Calculate MSE
        mse = mean((data_train - y_hat).^2);
        alpha_values(i) = alpha;
        mse_values(i) = mse;
        % Update minimum MSE and optimal alpha if a lower MSE is found
        if mse < min_mse
            min_mse = mse;
            optimal_alpha = alpha;
            optimal_mse = min_mse;
        end
    end
    % Plot alpha vs. MSE
    %plot(alpha_values, mse_values);
    %xlabel('Alpha');
    %ylabel('Mean Squared Error');
    %title('Alpha vs. Mean Squared Error');
    %grid on;
end

function [optimal_mse, optimal_alpha, optimal_beta, results] = reg(data)
    X = [ones(length(data(1:10)),1), (1:10)'];
    y=data(1:10);
    results=X\y;
    
    alphas = linspace(0, 1, 1001);
    betas = linspace(0, 1, 1001);
    
    % Initialize minimum MSE and corresponding alpha
    min_mse = inf;
    optimal_alpha = 0;
    optimal_beta = 0;
    y_hat = zeros(size(data));
    y_hat(1) = results(1);
    % Arrays to store alpha and corresponding MSE values
    alpha_values = zeros(size(alphas));
    beta_values = zeros(size(betas));
    mse_values = zeros(size(alphas));
    
    h = 1;
    l = results(1);
    b = results(2);

    for i = 1:length(alphas)
        alpha = alphas(i);

        for u = 1:length(betas)
            beta = betas(u);
        
            % Calculate predictions for the remaining data points
            for t = 2:length(data)
                l(t)=alpha*data(t)+(1-alpha)*(l(t-1)+b(t-1));
                b(t)= beta*(l(t)-l(t-1))+(1-beta)*b(t-1);
                y_hat(t) = l(t-1)+h*b(t-1);
            end

            mse = mean((data - y_hat).^2);
            alpha_values(i) = alpha;
            beta_values(u) = beta;
            mse_values(i, u) = mse;
            % Update minimum MSE and optimal alpha if a lower MSE is found
            if mse < min_mse
                min_mse = mse;
                optimal_alpha = alpha;
                optimal_beta = beta;
                optimal_mse = min_mse;
            end
        end
    end
end

function y_hat = predict(data, results, alpha, beta)
    l = results(1);
    b = results(2);
    h=1;
    y_hat = zeros(size(data));
    y_hat(1) = results(1);
    for t = 2:length(data)
        l(t)=alpha*data(t)+(1-alpha)*(l(t-1)+b(t-1));
        b(t)= beta*(l(t)-l(t-1))+(1-beta)*b(t-1);
        y_hat(t) = l(t-1)+h*b(t-1);
    end
end