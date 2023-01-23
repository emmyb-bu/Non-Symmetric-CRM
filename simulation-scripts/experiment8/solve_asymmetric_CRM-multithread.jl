#julia --threads nThreads
@time begin
nThreads = Threads.nthreads()
@info string(nThreads) * " threads available" 
import Pkg;
Pkg.add(["DifferentialEquations", "JLD2"])
@info "Packages installed"
using DifferentialEquations, JLD2
@info "Precompilation complete"
end

struct CRMParams
    c::Matrix{Float64}
    e::Matrix{Float64}
    K::Vector{Float64}
    m::Vector{Float64}
    S::Int64
    M::Int64
    CRMParams(c,e,K,m) = size(c,1)==size(m,1) && size(c,2)==size(K,1) && size(c) == size(e) ? new(c,e,K,m,size(m)[1],size(K)[1]) : error("parameters wrong dimensions")
end

function CRM!(du,u,p,t)
    @inbounds λ = @view u[1:p.S]
    @inbounds dλ = @view du[1:p.S]
    @inbounds R = @view u[end+1-p.M:end]
    @inbounds dR = @view du[end+1-p.M:end]
    dλ .= λ .* (p.c * R .- p.m)
    dR .= R .* (p.K .- R .- p.e' * λ)
end

M = 200;
S = 200;
μc = 1e0M;
μe = 1e0M;
σc = 2e-1sqrt(M);
σe = 2e-1sqrt(M);
K = 1e0;
σK = 1e-1;
m = 1e0;
σm = 1e-1;

@info "M = "*string(M)
@info "S = "*string(S)
@info "μc = "*string(μc/M)*"M"
@info "μe = "*string(μe/M)*"M"
@info "σc = "*string(σc/sqrt(M))*"√M"
@info "σe = "*string(σe/sqrt(M))*"√M"
@info "K = "*string(K)
@info "σK = "*string(σK)
@info "m = "*string(m)
@info "σm = "*string(σm)

function runCRM(ρ,tmax)
    K0 = K .+ σK*randn(M);
    params = randn(M,S) |> d -> CRMParams(μc/M .+ (σc/sqrt(M)).*d,μe/M .+ (σe/sqrt(M)).*(ρ.*d .+ sqrt(1-ρ^2).*randn(M,S)),K0,m .+ σm*randn(S))
    u0=vcat(ones(S)/S,K0)
    tspan=(0.0,tmax + 0.25);
    prob = ODEProblem(CRM!,u0,tspan,params);
    sol = solve(prob,Tsit5(),maxiters=Int64(5e7),abstol=1e-16,reltol=1e-16,saveat=(tmax-0.25):0.05:tmax+0.25);
    if sol.retcode != :Success
        println("Failure at ρ = "*string(ρ)*", thread "*string(Threads.threadid())*". Re-running...");
        sol = runCRM(ρ,tmax);
        # warning! bad statistical properties! can be avoided by using small tolerances...
    else
        return sol
    end
end

tmax = 9000;

ρs = 0.0:0.025:1.0

experiment_no = "8";

Threads.@threads for _ in 1:nThreads
    for ρ in ρs
        sols = [runCRM(ρ,tmax) for _ in 1:10];
        jldsave("./results/results_experiment"*experiment_no*"/experiment"*experiment_no*"-solution_ending_derivatives_thread="*string(Threads.threadid())*"_rho="*string(ρ)*".jld2";data = (X -> X(tmax,Val{1})).(sols));
        jldsave("./results/results_experiment"*experiment_no*"/experiment"*experiment_no*"-solution_ending_thread="*string(Threads.threadid())*"_rho="*string(ρ)*".jld2";data = (X -> X(tmax)).(sols));
    end
end
