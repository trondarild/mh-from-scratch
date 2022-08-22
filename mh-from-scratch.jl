#
#
using Random
using Distributions
d = Normal(10, 3)
mod1 = t -> rand(d, t)

population = mod1(30000)
observation = population[rand(1:30000, 1000)]

mu_obs = observation |> mean

transition_model = x -> [x[1], rand(Normal(x[2], 0.5), 1)]

function prior(x)
    #x[0] = mu, x[1]=sigma (new or current)
    #returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    #returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    #It makes the new sigma infinitely unlikely.
    if(x[2] <= 0) 
        return 0 
    end
    return 1
end
   
#Computes the likelihood of the data given a sigma (new or current) according to equation (2)
function manual_log_like_normal(x, data)
    #x[0]=mu, x[1]=sigma (new or current)
    #data = the observation
    return -log(x[2] * sqrt(2*pi)) .- ((data .- x[1]).^2) / (2*x[2]^2) |> sum
end

function acceptance(x, x_new)
    if x_new > x return true 
    else
        accept = rand(0:1)
        return (accept < (exp(x_new-x)))
    end
end

function metropolis_hastings(
    likelihood_computer,
    prior,
    transition_model,
    param_init,
    iterations,
    data,
    acceptance_rule)
    # likelihood_computer(x,data): returns the likelihood that these parameters generated the data
    # transition_model(x): a function that draws a sample from a symmetric distribution and returns it
    # param_init: a starting sample
    # iterations: number of accepted to generated
    # data: the data that we wish to model
    # acceptance_rule(x,x_new): decides whether to accept or reject the new sample 
    x = param_init
    accepted = []
    rejected = []
    for i in 1:iterations
        x_new = transition_model(x)
        x_lik = likelihood_computer(x, data)
        x_new_lik = likelihood_computer(x_new, data)
        if(acceptance_rule(x_lik + log(prior(x))),
            x_new_lik + log(prior(x_new)))
            x = x_new
            append!(accepted, x_new)
        else
            append!(rejected, x_new)
        end
    end
end

for i in 1:10 
    println(i) 
end

