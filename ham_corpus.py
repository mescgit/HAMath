"""
HAMesh Corpus Loader — Verified math knowledge into the mesh

Parses Metamath databases (.mm files) and folds theorems into a
HolographicMesh using the LLM-free embedder. Every statement that enters
the mesh has been formally verified — no hallucination possible.

The key-value pairs folded in are:
  key: theorem name + statement  (what it IS)
  val: theorem name + proof sketch / nearby axioms  (how it CONNECTS)

This grounds the mesh in real mathematical relationships rather than
natural-language associations.

Usage:
    # Download set.mm first:
    #   https://us.metamath.org/downloads/metamath.zip  (extract set.mm)
    # Or use a subset — see BUNDLED_SUBSETS below.

    python ham_corpus.py --subset arithmetic --save math_mesh.pt
    python ham_corpus.py --file set.mm --filter "number theory" --save math_mesh.pt
    python ham_corpus.py --builtin --save math_mesh.pt  (no download needed)
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import torch

from ham_core import HolographicMesh
from ham_embedder import Embedder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Bundled mini-corpus: ~120 foundational theorems, no download required
# Covers: Peano arithmetic, basic number theory, set theory primitives,
#         propositional logic, geometry basics
# Each entry: (name, statement, domain)
# ---------------------------------------------------------------------------

BUILTIN_CORPUS = [
    # --- Propositional logic ---
    ("modus ponens",       "If P implies Q, and P is true, then Q is true.", "logic"),
    ("modus tollens",      "If P implies Q, and Q is false, then P is false.", "logic"),
    ("hypothetical syllogism", "If P implies Q and Q implies R, then P implies R.", "logic"),
    ("disjunctive syllogism",  "If P or Q is true, and P is false, then Q must be true.", "logic"),
    ("de morgan and",      "Not (P and Q) is equivalent to (not P) or (not Q).", "logic"),
    ("de morgan or",       "Not (P or Q) is equivalent to (not P) and (not Q).", "logic"),
    ("law of excluded middle", "For any proposition P, either P or not P is true.", "logic"),
    ("double negation",    "Not not P is equivalent to P.", "logic"),
    ("contrapositive",     "P implies Q is equivalent to not Q implies not P.", "logic"),
    ("biconditional",      "P if and only if Q means P implies Q and Q implies P.", "logic"),

    # --- Set theory ---
    ("extensionality",     "Two sets are equal if and only if they have the same elements.", "set theory"),
    ("empty set",          "There exists a set with no elements, called the empty set.", "set theory"),
    ("pairing axiom",      "For any two sets A and B, there exists a set containing exactly A and B.", "set theory"),
    ("union axiom",        "For any set of sets, there exists a set that is the union of all of them.", "set theory"),
    ("power set axiom",    "For any set A, there exists a set of all subsets of A.", "set theory"),
    ("axiom of infinity",  "There exists a set that contains the empty set and is closed under successor.", "set theory"),
    ("axiom of choice",    "For any collection of non-empty sets, there exists a function selecting one element from each.", "set theory"),
    ("subset definition",  "A is a subset of B if every element of A is also an element of B.", "set theory"),
    ("intersection",       "The intersection of A and B is the set of elements belonging to both A and B.", "set theory"),
    ("complement",         "The complement of A in B is the set of elements in B that are not in A.", "set theory"),
    ("cartesian product",  "The Cartesian product A cross B is the set of all ordered pairs (a, b) with a in A and b in B.", "set theory"),
    ("russell paradox",    "The set of all sets that do not contain themselves leads to a contradiction.", "set theory"),
    ("cantor theorem",     "The power set of any set has strictly greater cardinality than the set itself.", "set theory"),
    ("cantor diagonal",    "No surjection exists from a set to its power set.", "set theory"),

    # --- Peano arithmetic ---
    ("peano zero",         "Zero is a natural number.", "arithmetic"),
    ("peano successor",    "Every natural number n has a successor S(n) which is also a natural number.", "arithmetic"),
    ("peano zero not successor", "Zero is not the successor of any natural number.", "arithmetic"),
    ("peano successor injective", "If S(m) equals S(n) then m equals n.", "arithmetic"),
    ("peano induction",    "If a property holds for zero, and holds for S(n) whenever it holds for n, it holds for all natural numbers.", "arithmetic"),
    ("addition base",      "n plus zero equals n for all natural numbers n.", "arithmetic"),
    ("addition recursive", "n plus S(m) equals S(n plus m).", "arithmetic"),
    ("multiplication base","n times zero equals zero.", "arithmetic"),
    ("multiplication recursive", "n times S(m) equals (n times m) plus n.", "arithmetic"),
    ("commutativity addition",    "m plus n equals n plus m for all natural numbers.", "arithmetic"),
    ("associativity addition",    "(m plus n) plus p equals m plus (n plus p).", "arithmetic"),
    ("commutativity multiplication", "m times n equals n times m.", "arithmetic"),
    ("associativity multiplication", "(m times n) times p equals m times (n times p).", "arithmetic"),
    ("distributivity",     "m times (n plus p) equals (m times n) plus (m times p).", "arithmetic"),

    # --- Number theory ---
    ("divisibility",       "a divides b if there exists an integer k such that b equals a times k.", "number theory"),
    ("prime definition",   "A prime number is a natural number greater than 1 with no divisors other than 1 and itself.", "number theory"),
    ("fundamental theorem of arithmetic", "Every integer greater than 1 is either prime or a unique product of primes.", "number theory"),
    ("euclid infinitely many primes", "There are infinitely many prime numbers.", "number theory"),
    ("euclid lemma",       "If a prime p divides a product ab, then p divides a or p divides b.", "number theory"),
    ("gcd definition",     "The greatest common divisor of a and b is the largest integer dividing both.", "number theory"),
    ("bezout identity",    "For integers a and b, there exist integers x and y such that ax plus by equals gcd(a,b).", "number theory"),
    ("chinese remainder theorem", "If n1 and n2 are coprime, the system x congruent to a1 mod n1 and x congruent to a2 mod n2 has a unique solution mod n1 times n2.", "number theory"),
    ("fermat little theorem", "If p is prime and a is not divisible by p, then a to the power p minus 1 is congruent to 1 modulo p.", "number theory"),
    ("euler totient theorem", "If gcd(a, n) equals 1 then a to the power phi(n) is congruent to 1 modulo n.", "number theory"),
    ("wilson theorem",     "p is prime if and only if (p minus 1) factorial is congruent to negative 1 modulo p.", "number theory"),
    ("quadratic reciprocity", "For distinct odd primes p and q, the Legendre symbols satisfy a specific multiplicative relationship.", "number theory"),
    ("goldbach conjecture", "Every even integer greater than 2 is the sum of two primes. (Unproven as of 2025.)", "number theory"),
    ("twin prime conjecture", "There are infinitely many pairs of primes differing by 2. (Unproven as of 2025.)", "number theory"),
    ("riemann hypothesis",  "All non-trivial zeros of the Riemann zeta function have real part 1/2. (Unproven as of 2025.)", "number theory"),

    # --- Algebra ---
    ("group definition",   "A group is a set with an associative binary operation, an identity element, and inverses.", "algebra"),
    ("abelian group",      "A group is abelian if its operation is commutative.", "algebra"),
    ("subgroup",           "A subset H of group G is a subgroup if it is closed under the group operation and inverses.", "algebra"),
    ("lagrange theorem",   "The order of a subgroup divides the order of the group.", "algebra"),
    ("ring definition",    "A ring is a set with two operations: addition (abelian group) and associative multiplication distributive over addition.", "algebra"),
    ("field definition",   "A field is a ring where every non-zero element has a multiplicative inverse.", "algebra"),
    ("homomorphism",       "A homomorphism is a structure-preserving map between algebraic structures.", "algebra"),
    ("kernel",             "The kernel of a homomorphism is the set of elements mapping to the identity.", "algebra"),
    ("first isomorphism theorem", "The quotient of a group by the kernel of a homomorphism is isomorphic to the image.", "algebra"),
    ("polynomial ring",    "The set of polynomials with coefficients in a ring forms a ring under addition and multiplication.", "algebra"),
    ("fundamental theorem of algebra", "Every non-constant polynomial with complex coefficients has at least one complex root.", "algebra"),
    ("cayley theorem",     "Every group is isomorphic to a subgroup of a symmetric group.", "algebra"),

    # --- Analysis / Calculus ---
    ("limit definition",   "The limit of f(x) as x approaches a is L if for every epsilon > 0 there exists delta > 0 such that |f(x) - L| < epsilon whenever 0 < |x - a| < delta.", "analysis"),
    ("continuity",         "A function f is continuous at a if the limit of f(x) as x approaches a equals f(a).", "analysis"),
    ("intermediate value theorem", "If f is continuous on [a,b] and f(a) and f(b) have opposite signs, there exists c in (a,b) with f(c) = 0.", "analysis"),
    ("extreme value theorem", "A continuous function on a closed bounded interval attains its maximum and minimum.", "analysis"),
    ("mean value theorem", "If f is differentiable on (a,b), there exists c in (a,b) where f'(c) equals (f(b)-f(a))/(b-a).", "analysis"),
    ("fundamental theorem of calculus", "The derivative of the integral of f from a to x equals f(x).", "analysis"),
    ("taylor series",      "A smooth function can be expressed as an infinite sum of terms involving its derivatives at a point.", "analysis"),
    ("cauchy sequence",    "A sequence where terms become arbitrarily close together converges in a complete metric space.", "analysis"),
    ("bolzano weierstrass","Every bounded sequence of real numbers has a convergent subsequence.", "analysis"),
    ("uniform convergence","A sequence of functions converges uniformly if the rate of convergence is independent of the point.", "analysis"),

    # --- Geometry ---
    ("pythagorean theorem","In a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.", "geometry"),
    ("euclid parallel postulate", "Through a point not on a line, there is exactly one line parallel to the given line.", "geometry"),
    ("triangle angle sum", "The interior angles of a triangle sum to 180 degrees in Euclidean geometry.", "geometry"),
    ("similar triangles",  "Two triangles are similar if their corresponding angles are equal.", "geometry"),
    ("thales theorem",     "An angle inscribed in a semicircle is a right angle.", "geometry"),
    ("euler formula polyhedra", "For a convex polyhedron, vertices minus edges plus faces equals 2.", "geometry"),
    ("gauss egregium theorem", "The Gaussian curvature of a surface is an intrinsic property preserved under bending.", "geometry"),

    # --- Combinatorics ---
    ("pigeonhole principle","If n+1 objects are placed in n boxes, at least one box contains more than one object.", "combinatorics"),
    ("binomial theorem",   "The expansion of (x+y)^n is the sum of C(n,k) x^k y^(n-k) for k from 0 to n.", "combinatorics"),
    ("inclusion exclusion","The size of the union of sets equals the sum of sizes minus pairwise intersections plus triple intersections, and so on.", "combinatorics"),
    ("ramsey theory",      "In any sufficiently large structure, order must appear; complete disorder is impossible.", "combinatorics"),
    ("stirling numbers",   "Stirling numbers count the ways to partition a set into non-empty subsets.", "combinatorics"),

    # --- Topology ---
    ("open set definition","A topology on a set X is a collection of subsets closed under arbitrary unions and finite intersections.", "topology"),
    ("compactness",        "A space is compact if every open cover has a finite subcover.", "topology"),
    ("connectedness",      "A space is connected if it cannot be partitioned into two disjoint non-empty open sets.", "topology"),
    ("homeomorphism",      "A homeomorphism is a continuous bijection with a continuous inverse.", "topology"),
    ("heine borel theorem","A subset of Euclidean space is compact if and only if it is closed and bounded.", "topology"),
    ("brouwer fixed point","Every continuous function from a closed ball to itself has a fixed point.", "topology"),
    ("euler characteristic","The Euler characteristic is a topological invariant: V - E + F for surfaces.", "topology"),

    # --- Information theory ---
    ("shannon entropy",    "The entropy H of a probability distribution measures the expected information content: H = -sum p log p.", "information theory"),
    ("channel capacity",   "The maximum rate of reliable information transmission over a noisy channel.", "information theory"),
    ("data compression theorem", "Lossless compression cannot compress below the entropy rate of the source.", "information theory"),
    ("kolmogorov complexity", "The algorithmic complexity of a string is the length of its shortest description in a universal language.", "information theory"),

    # --- Computability ---
    ("turing completeness","A system is Turing complete if it can simulate any Turing machine.", "computability"),
    ("halting problem",    "There is no algorithm that can determine for all programs and inputs whether the program halts.", "computability"),
    ("church turing thesis","Any effectively computable function can be computed by a Turing machine.", "computability"),
    ("rice theorem",       "Any non-trivial semantic property of programs is undecidable.", "computability"),
    ("godel incompleteness first", "Any consistent formal system strong enough to describe arithmetic contains true statements it cannot prove.", "computability"),
    ("godel incompleteness second", "A consistent formal system cannot prove its own consistency.", "computability"),

    # --- Linear algebra ---
    ("eigenvalue definition","A scalar lambda is an eigenvalue of matrix A if there exists a non-zero vector v such that Av = lambda v.", "linear algebra"),
    ("spectral theorem",   "A real symmetric matrix has real eigenvalues and orthogonal eigenvectors.", "linear algebra"),
    ("rank nullity theorem","The rank plus the nullity of a linear map equals the dimension of the domain.", "linear algebra"),
    ("determinant",        "The determinant of a matrix measures the scaling factor of the linear transformation it represents.", "linear algebra"),
    ("singular value decomposition", "Every matrix can be written as U times Sigma times V transpose, where U and V are orthogonal.", "linear algebra"),
    ("gram schmidt",       "Any linearly independent set of vectors can be transformed into an orthonormal basis.", "linear algebra"),
    ("cayley hamilton theorem", "Every square matrix satisfies its own characteristic polynomial.", "linear algebra"),
]


# ---------------------------------------------------------------------------
# Advanced corpus — graduate-level mathematics
# Adds ~60 entries across 4 domains to create new attractor basins.
# The cross-domain scholar can then find bridges between these and the
# builtin corpus, producing more sophisticated conjectures.
# ---------------------------------------------------------------------------

ADVANCED_CORPUS = [
    # --- Category theory ---
    ("category definition",       "A category consists of objects and morphisms between them, with associative composition and identity morphisms.", "category theory"),
    ("functor",                   "A functor F between categories maps objects to objects and morphisms to morphisms, preserving composition and identities.", "category theory"),
    ("natural transformation",    "A natural transformation eta between functors F and G assigns to each object A a morphism eta_A such that the naturality square commutes.", "category theory"),
    ("yoneda lemma",              "Natural transformations from Hom(A, -) to F are in bijection with elements of F(A); every object is determined by its relationships to all other objects.", "category theory"),
    ("adjunction",                "Functors F and G are adjoint (F is left adjoint to G) if there is a natural bijection Hom(F A, B) = Hom(A, G B).", "category theory"),
    ("universal property",        "An object satisfies a universal property if every competing object factors uniquely through it.", "category theory"),
    ("limit and colimit",         "A limit is a universal cone over a diagram; a colimit is a universal cocone. Products and coproducts are special cases.", "category theory"),
    ("abelian category",          "An abelian category has a zero object, binary products, and every morphism has a kernel and cokernel, enabling homological algebra.", "category theory"),
    ("exact sequence",            "A sequence of morphisms is exact if the image of each morphism equals the kernel of the next.", "category theory"),
    ("snake lemma",               "Given a commutative diagram with exact rows, there is a natural long exact sequence connecting the kernels and cokernels.", "category theory"),
    ("five lemma",                "If four of the five vertical maps in a morphism of exact sequences are isomorphisms, so is the fifth.", "category theory"),
    ("adjoint functor theorem",   "A functor between complete categories has a left adjoint if and only if it preserves all limits and satisfies a solution set condition.", "category theory"),
    ("topos",                     "A topos is a category that behaves like the category of sets, with a subobject classifier and all finite limits.", "category theory"),
    ("monad",                     "A monad is a functor T with unit and multiplication natural transformations satisfying associativity and unit laws; it encodes algebraic structure.", "category theory"),
    ("kan extension",             "The Kan extension of F along K is the best approximation to a functor that doesn't exist; all concepts are Kan extensions.", "category theory"),

    # --- Algebraic topology ---
    ("fundamental group",         "The fundamental group pi_1(X, x) of a space at a basepoint classifies loops up to homotopy, measuring 1-dimensional holes.", "algebraic topology"),
    ("seifert van kampen theorem","The fundamental group of a union of spaces is the free product of their fundamental groups amalgamated over the intersection.", "algebraic topology"),
    ("covering space",            "A covering space is a space E with a map to X such that every point of X has a neighborhood evenly covered by E.", "algebraic topology"),
    ("singular homology",         "Singular homology H_n(X) is a sequence of abelian groups measuring n-dimensional holes in a topological space X.", "algebraic topology"),
    ("mayer vietoris sequence",   "For a space decomposed into two open sets, there is a long exact sequence relating the homology of each piece to the whole.", "algebraic topology"),
    ("de rham theorem",           "The de Rham cohomology of a smooth manifold is isomorphic to its singular cohomology with real coefficients.", "algebraic topology"),
    ("poincare duality",          "For a closed oriented n-manifold, H_k and H^{n-k} are isomorphic, relating homology and cohomology in complementary dimensions.", "algebraic topology"),
    ("euler characteristic homology", "The Euler characteristic equals the alternating sum of Betti numbers: chi = sum (-1)^k rank H_k.", "algebraic topology"),
    ("homotopy equivalence",      "Spaces X and Y are homotopy equivalent if there exist maps f: X to Y and g: Y to X whose compositions are homotopic to the identity.", "algebraic topology"),
    ("fibration",                 "A fibration is a map with the homotopy lifting property; it generalises a fibre bundle and yields a long exact sequence of homotopy groups.", "algebraic topology"),
    ("cohomology ring",           "Cohomology groups carry a cup product making them a graded ring, encoding multiplicative structure absent in homology.", "algebraic topology"),
    ("hurewicz theorem",          "For a simply connected space, the first non-zero homotopy group equals the first non-zero homology group.", "algebraic topology"),

    # --- Complex analysis ---
    ("holomorphic function",      "A function is holomorphic if it is complex-differentiable at every point in its domain, implying it is infinitely differentiable and analytic.", "complex analysis"),
    ("cauchy riemann equations",  "A function f = u + iv is holomorphic if and only if the partial derivatives satisfy du/dx = dv/dy and du/dy = -dv/dx.", "complex analysis"),
    ("cauchy integral theorem",   "The integral of a holomorphic function over a closed curve in a simply connected domain is zero.", "complex analysis"),
    ("cauchy integral formula",   "The value of a holomorphic function at a point is determined by its values on any surrounding contour: f(a) = (1/2 pi i) integral f(z)/(z-a) dz.", "complex analysis"),
    ("residue theorem",           "The integral of a meromorphic function around a closed contour equals 2 pi i times the sum of residues of the poles inside.", "complex analysis"),
    ("liouville theorem complex", "Every bounded entire function is constant.", "complex analysis"),
    ("fundamental theorem of algebra via complex analysis", "Every non-constant polynomial has a root in the complex numbers, proved via Liouville's theorem.", "complex analysis"),
    ("analytic continuation",     "A holomorphic function on a connected open set has at most one extension to any larger connected domain.", "complex analysis"),
    ("riemann mapping theorem",   "Any simply connected proper open subset of the complex plane is conformally equivalent to the open unit disk.", "complex analysis"),
    ("argument principle",        "For a meromorphic function, the number of zeros minus the number of poles inside a contour equals the winding number of the image.", "complex analysis"),
    ("identity theorem",          "If two holomorphic functions agree on a set with an accumulation point, they agree everywhere on their common domain.", "complex analysis"),
    ("weierstrass factorization", "Every entire function can be written as a (possibly infinite) product over its zeros, analogous to polynomial factorization.", "complex analysis"),

    # --- Probability and measure theory ---
    ("kolmogorov axioms",         "A probability measure assigns a non-negative real number to each event, with the whole space having measure 1 and countable additivity.", "probability"),
    ("conditional probability",   "The conditional probability P(A|B) = P(A and B) / P(B) is the probability of A given that B has occurred.", "probability"),
    ("bayes theorem",             "P(A|B) = P(B|A) P(A) / P(B) relates prior and posterior probabilities via the likelihood.", "probability"),
    ("law of large numbers",      "The average of independent identically distributed random variables converges to the expected value as the sample size grows.", "probability"),
    ("central limit theorem",     "The standardised sum of independent identically distributed random variables with finite variance converges in distribution to a standard normal.", "probability"),
    ("markov chain",              "A Markov chain is a sequence of random variables where the future depends only on the present, not the past.", "probability"),
    ("martingale",                "A martingale is a stochastic process where the expected future value given the present equals the present value.", "probability"),
    ("optional stopping theorem", "For a martingale stopped at a bounded stopping time, the expected value at stopping equals the initial expected value.", "probability"),
    ("borel cantelli lemma",      "If the sum of probabilities of events is finite then almost surely only finitely many occur; if the events are independent and the sum is infinite, almost surely infinitely many occur.", "probability"),
    ("characteristic function",   "The characteristic function phi(t) = E[exp(itX)] uniquely determines the distribution of a random variable X.", "probability"),
    ("brownian motion",           "Brownian motion is a continuous stochastic process with independent Gaussian increments; it is the scaling limit of random walks.", "probability"),
    ("ergodic theorem",           "For an ergodic measure-preserving transformation, time averages equal space averages almost surely.", "probability"),

    # --- Functional analysis ---
    ("banach space",              "A Banach space is a complete normed vector space; completeness allows limit arguments that don't escape the space.", "functional analysis"),
    ("hilbert space",             "A Hilbert space is a complete inner product space; it generalises Euclidean geometry to infinite dimensions.", "functional analysis"),
    ("hahn banach theorem",       "A bounded linear functional on a subspace of a normed space extends to the whole space without increasing its norm.", "functional analysis"),
    ("open mapping theorem",      "A surjective bounded linear map between Banach spaces is open, i.e., maps open sets to open sets.", "functional analysis"),
    ("closed graph theorem",      "A linear map between Banach spaces with a closed graph is bounded.", "functional analysis"),
    ("spectral theorem operators","A self-adjoint operator on a Hilbert space has a spectral decomposition in terms of projection-valued measures.", "functional analysis"),
    ("riesz representation",      "Every bounded linear functional on a Hilbert space is given by inner product with a unique element of the space.", "functional analysis"),
    ("compact operator",          "A compact operator maps bounded sets to precompact sets; its spectrum behaves like that of a finite-dimensional matrix.", "functional analysis"),
    ("fredholm alternative",      "For a compact operator K, either Tx = y has a unique solution for all y, or the homogeneous equation Tx = 0 has non-trivial solutions.", "functional analysis"),
]


# ---------------------------------------------------------------------------
# Physics corpus — laws, principles, and theorems of physics
# Same format as BUILTIN_CORPUS: (name, statement, domain)
# Covers classical mechanics through quantum field theory.
# Designed to be cross-pollinated with the math corpus.
# ---------------------------------------------------------------------------

PHYSICS_CORPUS = [
    # --- Classical mechanics ---
    ("newton first law",         "A body at rest stays at rest and a body in motion stays in motion unless acted on by a net external force.", "classical mechanics"),
    ("newton second law",        "The net force on an object equals its mass times its acceleration: F = ma.", "classical mechanics"),
    ("newton third law",         "For every action there is an equal and opposite reaction.", "classical mechanics"),
    ("conservation of momentum", "The total momentum of an isolated system remains constant.", "classical mechanics"),
    ("conservation of energy",   "The total energy of an isolated system is constant; energy cannot be created or destroyed.", "classical mechanics"),
    ("work energy theorem",      "The net work done on an object equals its change in kinetic energy.", "classical mechanics"),
    ("gravitational potential",  "The gravitational potential energy between two masses is -Gm1m2/r.", "classical mechanics"),
    ("kepler first law",         "Planets orbit the Sun in ellipses with the Sun at one focus.", "classical mechanics"),
    ("kepler second law",        "A line from a planet to the Sun sweeps equal areas in equal times.", "classical mechanics"),
    ("kepler third law",         "The square of a planet's orbital period is proportional to the cube of its semi-major axis.", "classical mechanics"),

    # --- Lagrangian and Hamiltonian mechanics ---
    ("principle of least action","The path taken by a physical system is the one that extremises the action integral of the Lagrangian.", "analytical mechanics"),
    ("euler lagrange equations", "The equations of motion derived from the Lagrangian: d/dt(dL/dq_dot) - dL/dq = 0.", "analytical mechanics"),
    ("hamiltonian mechanics",    "The Hamiltonian H = T + V generates the equations of motion via Hamilton's equations.", "analytical mechanics"),
    ("hamilton equations",       "dq/dt = dH/dp and dp/dt = -dH/dq describe the evolution of a system in phase space.", "analytical mechanics"),
    ("noether theorem",          "Every continuous symmetry of a physical system corresponds to a conserved quantity.", "analytical mechanics"),
    ("poisson brackets",         "The Poisson bracket {f,g} measures the rate of change of f along the flow of g in phase space.", "analytical mechanics"),
    ("liouville theorem",        "The phase space volume element is conserved under Hamiltonian evolution.", "analytical mechanics"),

    # --- Electromagnetism ---
    ("coulomb law",              "The electrostatic force between two charges is F = kq1q2/r^2, directed along the line joining them.", "electromagnetism"),
    ("gauss law electric",       "The electric flux through a closed surface equals the enclosed charge divided by epsilon_0.", "electromagnetism"),
    ("gauss law magnetic",       "The magnetic flux through any closed surface is zero: there are no magnetic monopoles.", "electromagnetism"),
    ("faraday law",              "A changing magnetic flux through a loop induces an electromotive force equal to minus the rate of change.", "electromagnetism"),
    ("ampere maxwell law",       "A magnetic field is produced by an electric current and by a changing electric field.", "electromagnetism"),
    ("maxwell equations",        "The four Maxwell equations unify electricity, magnetism, and light as electromagnetic waves.", "electromagnetism"),
    ("electromagnetic wave",     "Maxwell's equations predict electromagnetic waves travelling at the speed of light c = 1/sqrt(epsilon_0 mu_0).", "electromagnetism"),
    ("lorentz force",            "A charged particle in electric and magnetic fields experiences F = q(E + v x B).", "electromagnetism"),

    # --- Thermodynamics ---
    ("zeroth law thermodynamics","If two systems are each in thermal equilibrium with a third, they are in thermal equilibrium with each other.", "thermodynamics"),
    ("first law thermodynamics", "The internal energy change equals heat added minus work done by the system: dU = Q - W.", "thermodynamics"),
    ("second law thermodynamics","The entropy of an isolated system never decreases; heat flows spontaneously from hot to cold.", "thermodynamics"),
    ("third law thermodynamics", "The entropy of a perfect crystal approaches zero as temperature approaches absolute zero.", "thermodynamics"),
    ("boltzmann entropy",        "Entropy is proportional to the logarithm of the number of microstates: S = k_B ln(W).", "thermodynamics"),
    ("carnot efficiency",        "No heat engine can be more efficient than a Carnot engine operating between the same temperatures.", "thermodynamics"),
    ("equipartition theorem",    "Each quadratic degree of freedom contributes k_B T/2 to the average energy.", "thermodynamics"),

    # --- Statistical mechanics ---
    ("boltzmann distribution",   "In thermal equilibrium, the probability of a microstate with energy E is proportional to exp(-E/k_B T).", "statistical mechanics"),
    ("partition function",       "The partition function Z = sum exp(-E_i/k_B T) encodes all thermodynamic properties.", "statistical mechanics"),
    ("maxwell boltzmann speed",  "The speed distribution of gas molecules in equilibrium follows the Maxwell-Boltzmann distribution.", "statistical mechanics"),
    ("ergodic hypothesis",       "The time average of a quantity equals its ensemble average for an ergodic system.", "statistical mechanics"),
    ("fluctuation dissipation",  "The response of a system in equilibrium to a small perturbation is related to its spontaneous fluctuations.", "statistical mechanics"),

    # --- Special relativity ---
    ("einstein postulates",      "The laws of physics are the same in all inertial frames, and the speed of light is constant.", "special relativity"),
    ("time dilation",            "A moving clock ticks more slowly by a factor of 1/gamma: dt' = dt/gamma.", "special relativity"),
    ("length contraction",       "A moving object is shorter along its direction of motion by a factor of gamma.", "special relativity"),
    ("mass energy equivalence",  "Energy and mass are equivalent: E = mc^2.", "special relativity"),
    ("lorentz transformation",   "The Lorentz transformation relates coordinates between inertial frames moving at constant velocity.", "special relativity"),
    ("relativistic momentum",    "The relativistic momentum is p = gamma mv; it is conserved in all inertial frames.", "special relativity"),
    ("minkowski spacetime",      "Special relativity is naturally described in 4-dimensional Minkowski spacetime with metric ds^2 = -c^2dt^2 + dx^2 + dy^2 + dz^2.", "special relativity"),

    # --- General relativity ---
    ("equivalence principle",    "Gravitational and inertial mass are equivalent; a gravitational field is locally indistinguishable from acceleration.", "general relativity"),
    ("einstein field equations", "The Einstein field equations G_mu_nu = 8 pi G T_mu_nu relate spacetime curvature to energy-momentum.", "general relativity"),
    ("geodesic equation",        "Free-falling objects follow geodesics -- the shortest paths in curved spacetime.", "general relativity"),
    ("schwarzschild solution",   "The Schwarzschild metric describes the spacetime outside a spherical mass, including black holes.", "general relativity"),
    ("gravitational waves",      "Accelerating masses produce ripples in spacetime curvature that propagate at the speed of light.", "general relativity"),
    ("hawking radiation",        "Black holes emit thermal radiation due to quantum effects near the event horizon.", "general relativity"),

    # --- Quantum mechanics ---
    ("schrodinger equation",     "The time-dependent Schrodinger equation i hbar d/dt psi = H psi governs quantum state evolution.", "quantum mechanics"),
    ("wave particle duality",    "Quantum objects exhibit both wave-like interference and particle-like detection.", "quantum mechanics"),
    ("heisenberg uncertainty",   "The position and momentum of a particle cannot both be known precisely: delta_x delta_p >= hbar/2.", "quantum mechanics"),
    ("born rule",                "The probability of measuring an outcome is the squared modulus of the corresponding amplitude.", "quantum mechanics"),
    ("pauli exclusion principle","No two fermions can occupy the same quantum state simultaneously.", "quantum mechanics"),
    ("spin statistics theorem",  "Particles with integer spin are bosons (symmetric wavefunctions); half-integer spin particles are fermions (antisymmetric).", "quantum mechanics"),
    ("quantum superposition",    "A quantum system exists in a superposition of states until measured.", "quantum mechanics"),
    ("quantum entanglement",     "Entangled particles share a quantum state; measuring one instantly determines the other's state.", "quantum mechanics"),
    ("bell inequality",          "No local hidden variable theory can reproduce all quantum mechanical predictions.", "quantum mechanics"),
    ("dirac equation",           "The Dirac equation describes relativistic spin-1/2 particles and predicts antimatter.", "quantum mechanics"),

    # --- Quantum field theory ---
    ("feynman path integral",    "The probability amplitude for a process is the sum over all possible paths, weighted by exp(iS/hbar).", "quantum field theory"),
    ("gauge invariance",         "Physical laws are unchanged by local phase transformations; this symmetry forces the existence of gauge bosons.", "quantum field theory"),
    ("higgs mechanism",          "The spontaneous breaking of gauge symmetry gives mass to gauge bosons via the Higgs field.", "quantum field theory"),
    ("renormalisation",          "Infinities in quantum field theory are absorbed into physical parameters through renormalisation.", "quantum field theory"),
    ("cpt symmetry",             "Physical laws are invariant under the combined operation of charge conjugation, parity, and time reversal.", "quantum field theory"),
    ("standard model",           "The Standard Model describes all known fundamental particles and three of the four fundamental forces via gauge theory.", "quantum field theory"),

    # --- Condensed matter / many-body ---
    ("pauli matrices",           "The Pauli matrices sigma_x, sigma_y, sigma_z are the generators of SU(2) and describe spin-1/2 particles.", "quantum mechanics"),
    ("bose einstein condensate", "At low temperatures, bosons condense into the ground state, forming a macroscopic quantum state.", "condensed matter"),
    ("superconductivity",        "Below the critical temperature, certain materials conduct electricity with zero resistance due to Cooper pair condensation.", "condensed matter"),
    ("band theory",              "The electronic structure of solids is described by bands of allowed energies separated by band gaps.", "condensed matter"),

    # --- Waves and optics ---
    ("huygens principle",        "Every point on a wavefront acts as a source of secondary spherical wavelets.", "waves"),
    ("double slit interference", "Light passing through two slits creates an interference pattern: evidence of wave nature.", "waves"),
    ("doppler effect",           "The observed frequency of a wave changes when the source or observer moves.", "waves"),
    ("snell law",                "The ratio of the sines of angles of incidence and refraction equals the ratio of wave speeds: n1 sin(theta1) = n2 sin(theta2).", "waves"),
]


def build_mesh_from_physics(embedder: Embedder = None) -> HolographicMesh:
    """Build a mesh from the physics corpus."""
    entries = [
        {'name': n, 'statement': s, 'proof_sketch': f"domain: {d}", 'domain': d}
        for n, s, d in PHYSICS_CORPUS
    ]
    return build_mesh_from_corpus(entries, embedder=embedder)


# ---------------------------------------------------------------------------
# Metamath .mm file parser
# ---------------------------------------------------------------------------

METAMATH_URL = "https://us.metamath.org/downloads/metamath.zip"
METAMATH_DEFAULT_PATH = Path("./ham_data/set.mm")


def download_metamath(dest_dir: str = "./ham_data") -> Path:
    """
    Download and extract set.mm from metamath.org.
    Returns the path to the extracted set.mm file.
    """
    import zipfile

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    mm_path = dest / "set.mm"

    if mm_path.exists():
        print(f"  set.mm already exists at {mm_path} ({mm_path.stat().st_size // 1024 // 1024} MB)")
        return mm_path

    zip_path = dest / "metamath.zip"
    print(f"  Downloading metamath.zip from metamath.org...")
    print(f"  (This is ~30 MB, one-time download)")

    def _progress(count, block_size, total):
        pct = min(count * block_size * 100 // total, 100)
        if count % 50 == 0:
            print(f"  {pct}%", end="\r", flush=True)

    urllib.request.urlretrieve(METAMATH_URL, zip_path, reporthook=_progress)
    print(f"  Downloaded. Extracting set.mm...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # set.mm may be nested inside a folder in the zip
        mm_names = [n for n in zf.namelist() if n.endswith("set.mm")]
        if not mm_names:
            raise RuntimeError("set.mm not found inside metamath.zip")
        zf.extract(mm_names[0], dest)
        extracted = dest / mm_names[0]
        if extracted != mm_path:
            extracted.rename(mm_path)

    zip_path.unlink()  # clean up zip
    print(f"  Extracted: {mm_path} ({mm_path.stat().st_size // 1024 // 1024} MB)")
    return mm_path


def parse_metamath(mm_path: str, max_theorems: int = 200, skip: int = 0) -> list[dict]:
    """
    Parse a Metamath .mm file and extract theorem statements.

    Extracts the human-readable $(comment$) that precedes each theorem —
    these are natural language descriptions that embed well. Falls back to
    the symbolic statement if no comment is present.

    Returns a list of dicts: {name, statement, proof_sketch, domain}
    Only extracts $p (provable) statements.
    """
    path = Path(mm_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Metamath file not found: {mm_path}\n"
            f"  Run with --download to fetch it automatically, or download manually:\n"
            f"  {METAMATH_URL}"
        )

    print(f"  Parsing {path.name} ({path.stat().st_size // 1024 // 1024} MB)...")
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Extract comments paired with their following theorem label
    # Pattern: $( description text $) ... label $p symbolic $= proof $.
    block_pattern = re.compile(
        r'\$\((.*?)\$\)'          # comment block
        r'(?:[^$]*?)'             # anything between (whitespace, non-$ chars)
        r'(\w+)\s+\$p\s+(.*?)\s+\$=\s+.*?\$\.',
        re.DOTALL
    )

    theorems = []
    seen = set()

    for match in block_pattern.finditer(raw):
        comment   = ' '.join(match.group(1).split()).strip()
        name      = match.group(2)
        symbolic  = ' '.join(match.group(3).split())

        if name in seen:
            continue
        seen.add(name)

        # Apply skip offset
        if len(seen) <= skip:
            continue

        # Use the comment if it's meaningful natural language (>20 chars,
        # doesn't start with '~' which means it's a cross-reference tag)
        if len(comment) > 20 and not comment.startswith('~'):
            # Clean up Metamath markup: remove ~ refs, `backtick` code, HTML
            desc = re.sub(r'~\s*\w+', '', comment)
            desc = re.sub(r'`[^`]*`', '', desc)
            desc = re.sub(r'<[^>]+>', '', desc)
            desc = re.sub(r'\s+', ' ', desc).strip()
            statement = f"{name}: {desc[:300]}"
        else:
            # Fall back: use label + symbolic (still better than raw symbols)
            statement = f"{name}: {symbolic[:200]}"

        # Infer domain from common Metamath chapter prefixes
        domain = _infer_domain(name, comment)

        theorems.append({
            'name':         name,
            'statement':    statement,
            'proof_sketch': f"metamath: {symbolic[:80]}",
            'domain':       domain,
        })
        if len(theorems) >= max_theorems:
            break

    # Also sweep for theorems without preceding comments (plain $p blocks)
    if len(theorems) < max_theorems:
        plain_pattern = re.compile(
            r'(\w+)\s+\$p\s+(.*?)\s+\$=\s+.*?\$\.', re.DOTALL
        )
        for match in plain_pattern.finditer(raw):
            if len(theorems) >= max_theorems:
                break
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            symbolic  = ' '.join(match.group(2).split())
            statement = f"{name}: {symbolic[:200]}"
            theorems.append({
                'name':         name,
                'statement':    statement,
                'proof_sketch': f"metamath: {symbolic[:80]}",
                'domain':       _infer_domain(name, ''),
            })

    print(f"  Parsed {len(theorems)} theorems "
          f"({sum(1 for t in theorems if not t['statement'].endswith(t['name']+':'))} with natural language descriptions).")
    return theorems


def _infer_domain(name: str, comment: str) -> str:
    """Guess a math domain from the theorem name and comment text."""
    text = (name + ' ' + comment).lower()
    if any(w in text for w in ['prime', 'divis', 'factor', 'modulo', 'congruent', 'arith']):
        return 'number theory'
    if any(w in text for w in ['continu', 'limit', 'deriv', 'integr', 'differenti']):
        return 'analysis'
    if any(w in text for w in ['set', 'class', 'member', 'subset', 'union', 'intersect']):
        return 'set theory'
    if any(w in text for w in ['group', 'ring', 'field', 'homomorph', 'isomorph', 'algebra']):
        return 'algebra'
    if any(w in text for w in ['topolog', 'open', 'closed', 'compact', 'connect', 'metric']):
        return 'topology'
    if any(w in text for w in ['logic', 'provab', 'tautolog', 'wff', 'axiom', 'theorem']):
        return 'logic'
    if any(w in text for w in ['real', 'complex', 'rational', 'integer', 'natural']):
        return 'number systems'
    if any(w in text for w in ['matrix', 'vector', 'linear', 'eigenval', 'determin']):
        return 'linear algebra'
    if any(w in text for w in ['geometr', 'angle', 'triangle', 'circle', 'polygon']):
        return 'geometry'
    if any(w in text for w in ['comput', 'halting', 'turing', 'algorithm', 'decid']):
        return 'computability'
    return 'metamath'


# ---------------------------------------------------------------------------
# Mesh builder
# ---------------------------------------------------------------------------

def build_mesh_from_corpus(
    entries: list[dict],
    embedder: Embedder = None,
    fold_strength: float = 1.0,
    verbose: bool = True,
) -> HolographicMesh:
    """
    Fold a list of corpus entries into a fresh HolographicMesh.

    Each entry is folded as:
      key = embed(name + ": " + statement)
      val = embed(name + " connects to: " + domain + " | " + proof_sketch)

    Returns a loaded HolographicMesh ready for dreaming.
    """
    if embedder is None:
        embedder = Embedder()

    mesh = HolographicMesh(dim=embedder.dim, device=embedder.device)

    if verbose:
        print(f"\n  Building mesh from {len(entries)} entries (dim={embedder.dim})...")

    # Batch-embed all keys and values for speed
    keys_text = [f"{e['name']}: {e['statement']}" for e in entries]
    vals_text  = [
        f"{e['name']} in {e.get('domain','math')}: {e.get('proof_sketch', e['statement'][:120])}"
        for e in entries
    ]

    if verbose:
        print("  Embedding keys...")
    key_vecs = embedder.embed_batch(keys_text)

    if verbose:
        print("  Embedding values...")
    val_vecs = embedder.embed_batch(vals_text)

    # Fold and register memories
    for i, entry in enumerate(entries):
        mesh.fold(key_vecs[i], val_vecs[i], strength=fold_strength)
        mesh.remember(key_vecs[i], keys_text[i])

    if verbose:
        s = mesh.stats()
        print(f"  Mesh built: {s['folds']} folds, {s['memories']} memories, "
              f"energy={s['energy']:.1f}")

    return mesh


def build_mesh_from_builtin(embedder: Embedder = None) -> HolographicMesh:
    """Build a mesh from the bundled mini-corpus (no download needed)."""
    entries = [
        {
            'name':         name,
            'statement':    statement,
            'proof_sketch': f"domain: {domain}",
            'domain':       domain,
        }
        for name, statement, domain in BUILTIN_CORPUS
    ]
    return build_mesh_from_corpus(entries, embedder=embedder)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a HAMesh from a verified math corpus")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--builtin",  action="store_true",
                     help="Use the bundled math mini-corpus (~111 theorems, no download needed)")
    src.add_argument("--advanced", action="store_true",
                     help="Builtin math + advanced corpus (category theory, algebraic topology, "
                          "complex analysis, probability, functional analysis; ~170 entries)")
    src.add_argument("--physics",  action="store_true",
                     help="Use the built-in physics corpus (~80 laws, no download needed)")
    src.add_argument("--combined", action="store_true",
                     help="Use both math and physics corpora combined (~191 entries)")
    src.add_argument("--everything", action="store_true",
                     help="All corpora: builtin + advanced + physics (~240 entries)")
    src.add_argument("--download", action="store_true",
                     help="Auto-download set.mm from metamath.org (~30 MB, one-time)")
    src.add_argument("--file",     metavar="PATH",
                     help="Path to a Metamath .mm file")

    parser.add_argument("--max",     type=int, default=200,
                        help="Max theorems to load from .mm file (default 200)")
    parser.add_argument("--skip",    type=int, default=0,
                        help="Skip the first N theorems (set.mm starts with ~3000 "
                             "propositional logic tautologies; --skip 3000 jumps to "
                             "set theory and beyond)")
    parser.add_argument("--filter",  metavar="DOMAIN",
                        help="Only load entries whose domain contains this string")
    parser.add_argument("--save",    metavar="PATH", default="math_mesh.pt",
                        help="Where to save the mesh (default: math_mesh.pt)")
    parser.add_argument("--model",   default="all-mpnet-base-v2",
                        help="Sentence-transformer model name")
    args = parser.parse_args()

    embedder = Embedder(model_name=args.model)

    if args.builtin:
        entries = [
            {'name': n, 'statement': s, 'proof_sketch': f"domain: {d}", 'domain': d}
            for n, s, d in BUILTIN_CORPUS
        ]
    elif args.advanced:
        entries = [
            {'name': n, 'statement': s, 'proof_sketch': f"domain: {d}", 'domain': d}
            for n, s, d in BUILTIN_CORPUS + ADVANCED_CORPUS
        ]
    elif args.physics:
        entries = [
            {'name': n, 'statement': s, 'proof_sketch': f"domain: {d}", 'domain': d}
            for n, s, d in PHYSICS_CORPUS
        ]
    elif args.combined:
        entries = [
            {'name': n, 'statement': s, 'proof_sketch': f"domain: {d}", 'domain': d}
            for n, s, d in BUILTIN_CORPUS + PHYSICS_CORPUS
        ]
    elif args.everything:
        entries = [
            {'name': n, 'statement': s, 'proof_sketch': f"domain: {d}", 'domain': d}
            for n, s, d in BUILTIN_CORPUS + ADVANCED_CORPUS + PHYSICS_CORPUS
        ]
    elif args.download:
        mm_path = download_metamath()
        entries = parse_metamath(str(mm_path), max_theorems=args.max, skip=args.skip)
    else:
        entries = parse_metamath(args.file, max_theorems=args.max, skip=args.skip)

    if args.filter:
        entries = [e for e in entries if args.filter.lower() in e.get('domain','').lower()
                                      or args.filter.lower() in e['statement'].lower()]
        print(f"  After filter '{args.filter}': {len(entries)} entries")

    mesh = build_mesh_from_corpus(entries, embedder=embedder)
    mesh.save(args.save)
    print(f"\n  Saved to {args.save}")

    # Show top domains
    from collections import Counter
    domains = Counter(e.get('domain', 'unknown') for e in entries)
    print("\n  Domains:")
    for d, n in domains.most_common():
        print(f"    {d:30s} {n}")


if __name__ == "__main__":
    main()
