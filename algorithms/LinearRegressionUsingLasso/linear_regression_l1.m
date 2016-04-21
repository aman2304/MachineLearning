function [ w ] = linear_regression_l1(X, y, lambd, tolerance, maxIter, w0)
% LINEAR_REGRESSION_L1  Performs simple linear regression on samples in X
% and y, penalizing weights for more sparsity using lasso.
%   w = LINEAR_REGRESSION_L1(X, y, 0.1, 1e-6, 1000, zeros(size(X,2),1))
%   performs linear regression on X and y using the provided parameters
    
    w = w0;
    err = inf;
    iter_number = 1;
    beta = 0.75; % Shrinkage parameter for backtracking line search
    obj_values = zeros(maxIter, 1); % Storing objective values across iterations
    
    while err>=tolerance && iter_number<maxIter
        %% Computer gradient and step-size for proximal gradient descent
        grad = compute_gradient(X, y, w);
        step_size = line_search(X, y, w, grad, lambd, beta);
        
        %% Perform soft-thresholding on gradient iterate, re-evaluate objective 
        w = wthresh(w - step_size * grad, 's', lambd * step_size);
        obj_values(iter_number) = compute_objective(X, y, w, lambd, iter_number);
        
        %% Increment iteration number and relative error for convergence
        if(iter_number>1)
            err = abs(obj_values(iter_number) - obj_values(iter_number-1))/abs(obj_values(iter_number));
        end
        iter_number = iter_number + 1;
    end

end


function [gradient] = compute_gradient(X, y, w)
    gradient = -2*X'*y + 2*(X'*X)*w;
end

function [objective_value] = compute_objective(X, y, w, lambd, iter)
    objective_value = (y-X*w)' * (y-X*w) + lambd*norm(w,1);
    disp(['Value of objective function at iteration ' num2str(iter) ' is ' num2str(objective_value) '']);
end

function [t] = line_search(X, y, w, grad, lambd, beta)
    % Performs backtracking line search to get step size
    t = 1;
    while true
        generalized_grad = (w - wthresh(w - t * grad, 's', lambd * t))/(t);
        lhs = (y-X*(w - t*generalized_grad))' * (y-X*(w - t*generalized_grad));
        rhs = (y-X*w)' * (y-X*w) ...
              - t * grad' * generalized_grad ...
              + 0.5 * t * norm(generalized_grad,2) * norm(generalized_grad,2);
        if(lhs<=rhs)
            break;
        end
        t =  t * beta;        
    end
end