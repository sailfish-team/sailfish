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

	*a1 *= ${cex(1.0 / 2.0)};
	*a2 *= ${cex(-1.0 / 6.0)};
	*a3 *= ${cex(1.0 / 12.0)};
	*a4 *= ${cex(-1.0 / 20.0)};
}

// Estimates the alpha relaxation parameter using an asymptotic expansion
// of the entropy equality H(f) = H(f^alpha) in powers of fneq / f, where
// f^alpha is the entropic mirror state f + alpha * fneq.
// fneq = feq - fi
${device_func} inline float EstimateAlphaSeries(Dist* fi, Dist* fneq) {
	float a1, a2, a3, a4;
	ComputeACoeff(fi, fneq, &a1, &a2, &a3, &a4);
	return ${cex(sym.alpha_series())};
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

// Calculates entropy for the mirror state with a given alpha (f^alpha = f + alpha * fneq)
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

// Returns the maximum value of alpha for which all components of f_alpha are positive.
${device_func} inline float FindMaxAlpha(Dist* fi, Dist *fneq) {
	float max_alpha = 1000.0f;

	%for name in grid.idx_name:
		if (fi->${name} < 0.0f || fneq->${name} < 0.0f) {
			max_alpha = min(max_alpha, -fi->${name} / fneq->${name});
		}
	%endfor

	return max_alpha;
}

${device_func} inline float EstimateAlphaFromEntropy(Dist* fi, Dist* fneq, float alpha) {
	float ent = CalculateEntropy(fi);
	int i = 0;
	float init_alpha = alpha;
	const float max_alpha = FindMaxAlpha(fi, fneq);

	// Newton's method to find alpha.
	while (true) {
		float delta_ent_derivative;
		float ent_ineq = CalculateEntropyIneq(fi, fneq, alpha, &delta_ent_derivative);
		if (isnan(ent_ineq) && alpha != 1.1f) {
			alpha = 1.1f;
			continue;
		}
		float ent_increase = ent_ineq - ent;
		if (fabsf(ent_increase) < ${cex(entropy_tolerance)}) {
			break;
		}
		// Newton's method to solve: H(f_i) = H(f + alpha f_neq).
		float new_alpha = alpha - ent_increase / delta_ent_derivative;

		if (new_alpha > max_alpha) {
			// Fall back to the middle of the interval in case Newton's
			// method would result in an invalid alpha.
			new_alpha = 0.5f * (alpha + max_alpha);
		}

		if (fabsf(new_alpha - alpha) < ${cex(alpha_tolerance)}) {
			break;
		}

		alpha = new_alpha;
		i++;
		if (i > 1000) {
			%if gpu_check_invalid_values:
				${dump_dists('(*fi)', short=True)}
				${dump_dists('(*fneq)', short=True)}
				printf("Alpha estimation did not converge after %d iterations. alpha=%e H=%e dH=%e."
						" init=%e max=%e\n", i, alpha, ent, delta_ent_derivative, init_alpha, max_alpha);
			%endif
			die();
		}
	}

	if (alpha < 1.0f || !isfinite(alpha)) {
		%if gpu_check_invalid_values:
			${dump_dists('(*fi)', short=True)}
			${dump_dists('(*fneq)', short=True)}
			printf("Alpha estimated at: %e\n", alpha);
		%endif
		die();
	}
	return alpha;
}

// Returns true if the deviation of distribution from the equilibrium is small
// enough so that the asymptotic alpha expansion can be applied (see EstimateAlphaSeries).
${device_func} inline float SmallEquilibriumDeviation(Dist* fi, Dist* fneq) {
	float deviation = 0.0f;
	float t;

	%for name in grid.idx_name:
		t = fabsf(fneq->${name} / fi->${name});
		deviation = max(deviation, t);
		if (deviation > 0.01f) {
			return deviation;
		}
	%endfor
	return deviation;
}

${device_func} inline float EntropicRelaxationParam(Dist* fi, Dist* fneq
		${cond(alpha_output, ', ' + global_ptr + ' float* alpha_out')}) {
	const float dev = SmallEquilibriumDeviation(fi, fneq);
	float alpha;
	if (dev < 1e-6f) {
		alpha = 2.0f;
	} else if (dev < 0.01f) {
		alpha = EstimateAlphaSeries(fi, fneq);
	} else {
		%if alpha_output:
			alpha = EstimateAlphaFromEntropy(fi, fneq, *alpha_out);
		%else:
			alpha = EstimateAlphaFromEntropy(fi, fneq, 2.0f);
		%endif
	}

	%if alpha_output:
		// Always save alpha in global memory so that it can be used as a starting
		// point for the Newton-Rhapson method in the next iteration.
		*alpha_out = alpha;
	%endif

	return alpha;
}
