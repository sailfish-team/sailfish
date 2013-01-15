<%!
	from sailfish import sym
	from math import log
%>

<%namespace file="code_common.mako" import="*"/>

// See PRL 97, 010201 (2006), Eq. 12 for form of coefficients.
${device_func} inline void ComputeACoeff(Dist* fi, Dist* fneq, float *a1,
		float *a2, float *a3, float *a4) {
	*a1 = 0.0f;
	*a2 = 0.0f;
	*a3 = 0.0f;
	*a4 = 0.0f;
	float t, p;

	%for name in grid.idx_name:
		t = fneq->${name};
		p = t * t / fi->${name};
		t /= fi->${name};
		*a1 += p;
		p = p * t;
		*a2 += p;
		p = p * t;
		*a3 += p;
		p = p * t;
		*a4 += p;
	%endfor

	*a1 *= 1.0f / 2.0f;
	*a2 *= -1.0f / 6.0f;
	*a3 *= 1.0f / 12.0f;
	*a4 *= -1.0f / 20.0f;
}

// Estimates the alpha relaxation parameter using an asymptotic expansion
// of the entropy equality H(f) = H(f^alpha) in powers of fneq / f, where
// f^alpha is the entropic mirror state f + alpha * fneq.
// fneq = feq - fi
${device_func} inline float EstimateAlphaSeries(Dist* fi, Dist* fneq) {
	float a1, a2, a3, a4;

	ComputeACoeff(fi, fneq, &a1, &a2, &a3, &a4);
	float alpha = ${cex(sym.alpha_series())};
	return alpha;
}

// TODO(mjanusz): The entropy calculation should use a sympy expression.
// Calculates the entropy in the ELBM model.
${device_func} inline float CalculateEntropy(Dist* fi) {
	float ent = 0.0f;

	%for w, name in zip(grid.entropic_weights, grid.idx_name):
		ent += fi->${name} * (logf(fi->${name}) + (${cex(-log(w))}));
	%endfor

	return ent;
}

// Calculates entropy for the mirror state with a given alpha.
// Also calculates d \Delta entropy / d alpha and returns it in 'derivative'.
${device_func} inline float CalculateEntropyIneq(Dist* fi, Dist* fneq, float alpha, float *derivative) {
	float ent = 0.0f;
	float dent = 0.0f;
	float t, h;

	%for w, name in zip(grid.entropic_weights, grid.idx_name):
		t = fi->${name} + alpha * fneq->${name};
		h = logf(t) + (${cex(-log(w))});
		ent += t * h;
		dent += fneq->${name} * (h + 1.0f);
	%endfor

	*derivative = dent;
	return ent;
}

${device_func} inline float EstimateAlphaFromEntropy(Dist* fi, Dist* fneq, float alpha) {
	float ent = CalculateEntropy(fi);
	int i = 0;

	// Newton's method to find alpha.
	while (true) {
		float delta_ent_derivative;
		float ent_ineq = CalculateEntropyIneq(fi, fneq, alpha, &delta_ent_derivative);
		float ent_increase = ent_ineq - ent;
		if (ent_increase < ${cex(entropy_tolerance)}) {
			break;
		}
		// Newton's method to solve: H(f_i) = H(f + alpha f_neq).
		alpha = alpha - ent_increase / delta_ent_derivative;
		i++;
		if (i > 10000) {
			%if gpu_check_invalid_values:
				printf("Alpha estimation did not converge after %d iterations. alpha=%e H=%e dH=%e\n", i, alpha, ent, delta_ent_derivative);
			%endif
			die();
		}
	}

	if (alpha < 1.0f || !isfinite(alpha)) {
		%if gpu_check_invalid_values:
			printf("Alpha estimated at: %e\n", alpha);
		%endif
		die();
	}
	return alpha;
}

// Returns true if the deviation of distribution from the equilibrium is small
// enough so that the asymptotic alpha expansion can be applied (see EstimateAlphaSeries).
${device_func} inline bool SmallEquilibriumDeviation(Dist* fi, Dist* feq) {
	%for i in range(grid.Q):
		if (fabsf((feq->${grid.idx_name[i]} - fi->${grid.idx_name[i]}) / fi->${grid.idx_name[i]}) > 0.01f) {
			return false;
		}
	%endfor
	return true;
}
