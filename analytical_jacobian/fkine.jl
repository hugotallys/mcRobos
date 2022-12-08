using Symbolics

@variables d0 θ1 θ2 a2 θ3 a3 a4 

ξB0 = [1 0 0 0; 0 1 0 0; 0 0 1 d0; 0 0 0 1]
ξ01 = [cos(θ1) 0 -sin(θ1) 0; sin(θ1) 0 -cos(θ1) 0; 0 -1 0 0; 0 0 0 1]
ξ12 = [cos(θ2) -sin(θ2) 0 a2*cos(θ2); sin(θ2) cos(θ2) 0 a2*sin(θ2); 0 0 1 0; 0 0 0 1]
ξ23 = [cos(θ3) -sin(θ3) 0 a3*cos(θ3); sin(θ3) cos(θ3) 0 a3*sin(θ3); 0 0 1 0; 0 0 0 1]
ξ34 = [0 0 1 0; 1 0 0 -a4; 0 1 0 0; 0 0 0 1]

ξBE = ξB0 * ξ01 * ξ12 * ξ23 * ξ34

fXθ = ξBE[1, 4]
fYθ = ξBE[2, 4]
fZθ = ξBE[3, 4]

Dθ1 = Differential(θ1)
Dθ2 = Differential(θ2)
Dθ3 = Differential(θ3)

Jθ = [
    Dθ1(fXθ) Dθ2(fXθ) Dθ3(fXθ);
    Dθ1(fYθ) Dθ2(fYθ) Dθ3(fYθ);
    Dθ1(fZθ) Dθ2(fZθ) Dθ3(fZθ);
]

println("--- Analytical Jacobian ---\n--- Row 1 ---\nfX(θ) =")
display(fXθ)
println("\ndfX(θ)/dθ1 = ")
display(expand_derivatives(Jθ[1, 1]))
println("\ndfX(θ)/dθ2 = ")
display(expand_derivatives(Jθ[1, 2]))
println("\ndfX(θ)/dθ3 = ")
display(expand_derivatives(Jθ[1, 3]))

println("\n--- Row 2 ---\nfY(θ) =")
display(fYθ)
println("\ndfY(θ)/dθ1 = ")
display(expand_derivatives(Jθ[2, 1]))
println("\ndfY(θ)/dθ2 = ")
display(expand_derivatives(Jθ[2, 2]))
println("\ndfY(θ)/dθ3 = ")
display(expand_derivatives(Jθ[2, 3]))

println("\n--- Row 3 ---\nfZ(θ) =")
display(fZθ)
println("\ndfZ(θ)/dθ1 = ")
display(expand_derivatives(Jθ[3, 1]))
println("\ndfZ(θ)/dθ2 = ")
display(expand_derivatives(Jθ[3, 2]))
println("\ndfZ(θ)/dθ3 = ")
display(expand_derivatives(Jθ[3, 3]))
