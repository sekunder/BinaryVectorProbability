using BinaryVectorProbability
using Base.Test

X = rand(10,10000); p = collect(0.2:0.05:0.65); X = X .< p;
P = first_order_model(X); P2 = first_order_model(X);
P3 = BernoulliCodeDistribution(P.p)

@test isequal(P, P2)
@test !isequal(P, P3)

@test hash(P) == hash(P2)
@test hash(P) != hash(P3)
