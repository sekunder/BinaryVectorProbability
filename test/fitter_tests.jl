using BinaryVectorProbability
using DataFrames

Base.Random.srand(170017)

# N_neurons = 10
# Jtrue = rand(N_neurons, N_neurons); Jtrue = (Jtrue + Jtrue')/2
# P2 = IsingDistribution(Jtrue; comment="true distribution")

#testing if i can recover moments properly
neuron_numbers = 10:5:20
N_trials = 10
df = DataFrame(
    N_neurons=Int[],
    N_samples=Int[],
    trial=Int[],
    Jdiff_MPF=Float64[], Jdiff_LL=Float64[],
    thetadiff_MPF=Float64[], thetadiff_LL=Float64[],
    mudiff_MPF=Float64[], mudiff_LL=Float64[],
    sample_time=Float64[],
    fit_time_MPF=Float64[], fit_time_LL=Float64[])
Ptrue = Vector{IsingDistribution}(length(neuron_numbers))
mutrue = similar(Ptrue, Matrix{Float64})
Phat_MPF = Array{IsingDistribution,2}(length(neuron_numbers), N_trials)
Phat_LL = similar(Phat_MPF)
muhat_MPF = similar(Phat_MPF, Matrix{Float64})
muhat_LL = similar(muhat_MPF)
for (idx, N_neurons) in enumerate(neuron_numbers)
    Jtrue = 10 * randn(N_neurons, N_neurons); Jtrue = (Jtrue + Jtrue')/2
    Ptrue[idx] = IsingDistribution(Jtrue)
    mutrue[idx] = expectation_matrix(Ptrue[idx])
    for N_samples in [100N_neurons, 1000N_neurons, 5000N_neurons]
        for trial = 1:N_trials
            println("*** $N_neurons neurons, $N_samples samples, trial #$trial")
            print("  * Sampling...")
            tic()
            X = random(Ptrue[idx], N_samples)
            sample_time = toq()
            println("done: $sample_time s")

            print("  * Fitting using MPF...")
            tic()
            Phat_MPF[idx, trial] = second_order_model(X, verbose=0, fun="MPF")
            fit_time_MPF = toq()
            println("done: $fit_time_MPF s")

            print("  * Fitting using LL...")
            tic()
            Phat_LL[idx, trial] = second_order_model(X, verbose=0, fun="loglikelihood")
            fit_time_LL = toq()
            println("done:  $fit_time_LL s")

            muhat_MPF[idx, trial] = expectation_matrix(Phat_MPF[idx,trial])
            muhat_LL[idx, trial] = expectation_matrix(Phat_LL[idx,trial])

            Jdiff_MPF = norm(Phat_MPF[idx, trial].J - Ptrue[idx].J) / norm(Ptrue[idx].J)
            thetadiff_MPF = norm(Phat_MPF[idx, trial].theta - Ptrue[idx].theta) / norm(Ptrue[idx].theta)
            mudiff_MPF = norm(muhat_MPF[idx,trial] - mutrue[idx]) / norm(mutrue[idx])
            # maxmu_MPF = maximum(abs.((muhat_MPF[idx,trial] - mutrue[idx]) ./ mutrue[idx]))

            Jdiff_LL = norm(Phat_LL[idx, trial].J - Ptrue[idx].J) / norm(Ptrue[idx].J)
            thetadiff_LL = norm(Phat_LL[idx, trial].theta - Ptrue[idx].theta) / norm(Ptrue[idx].theta)
            mudiff_LL = norm(muhat_LL[idx,trial] - mutrue[idx]) / norm(mutrue[idx])
            # maxmu_LL = maximum(abs.((muhat_LL[idx,trial] - mutrue[idx]) ./ mutrue[idx]))

            # println(">>> Sampling:     $sample_time s")
            # println(">>> Fitting:      $fit_time s")
            println("  * |ΔJ/J| (MPF)  $Jdiff_MPF")
            println("  * |Δθ/θ| (MPF)  $thetadiff_MPF")
            println("  * |Δμ/μ| (MPF)  $mudiff_MPF")
            # println(">>> max Δμ/μ  $maxmu")
            println()
            println("  * |ΔJ/J| (LL)   $Jdiff_LL")
            println("  * |Δθ/θ| (LL)   $thetadiff_LL")
            println("  * |Δμ/μ| (LL)   $mudiff_LL")

            println()
            push!(df, [N_neurons,
                        N_samples,
                        trial,
                        Jdiff_MPF, Jdiff_LL,
                        thetadiff_MPF, thetadiff_LL,
                        mudiff_MPF, mudiff_LL,
                        sample_time,
                        fit_time_MPF, fit_time_LL])
        end
    end
end

# testing if I can fit data keeping H_1 > H_2 > H_n
using Spikes, Optim
Base.Random.srand(170017)

funcs = [t -> 1 + k + (k/2) * sin(t + pi/k) + (k/2) * cos(t/k) for k = 1:10]
TT = [inhomogeneous_poisson_process(f.(0:0.01:3000), 0.01) for f in funcs]
Sp = SpikeTrains(TT)
X = transpose(raster(Sp, 0.020))

P_1 = first_order_model(X)
P_2_MPF = second_order_model(X; verbose=2)
J0 = P_2_MPF.metadata[:J0]
P_2_L = second_order_model(X; verbose=2, allow_f_increases=true, J0=J0)
P_2_different_m = second_order_model(X; verbose=2, J0=J0, algorithm=LBFGS(m=20))
P_2_different_m_increase = second_order_model(X; verbose=2, J0=J0, algorithm=LBFGS(m=20), allow_f_increases=true)
P_2_large_m = second_order_model(X; verbose=2, J0=J0, algorithm=LBFGS(m=40))
P_2_thirdtime = second_order_model(X; verbose=2)
P_2_fourthtime = second_order_model(X; verbose=2)
P_N = data_model(X)

H_1 = entropy2(P_1)
H_2_MPF = entropy2(P_2_MPF)
H_2_L = entropy2(P_2_L)
H_2_thirdtime = entropy2(P_2_thirdtime)
H_2_fourthtime = entropy(P_2_fourthtime)
H_N = entropy2(P_N)

mu_1 = expectation_matrix(P_1)
mu_2_MPF = expectation_matrix(P_2_MPF)
mu_2_L = expectation_matrix(P_2_L)
mu_N = expectation_matrix(P_N)

for P in [P_2_MPF, P_2_L, P_2_thirdtime, P_2_fourthtime]
    println("convergence: $(Optim.converged(P.metadata[:opt_res]))\tentropy in bounds: $(H_N < entropy2(P) < H_1)")
end
