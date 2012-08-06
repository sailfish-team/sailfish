<%!
	from sailfish import sym
%>

<%namespace file="code_common.mako" import="*"/>

${device_func} inline void ComputeACoeff(Dist* fi, Dist* feq, float *a1,
		float *a2, float *a3, float *a4) {
	*a1 = 0.0f;
	*a2 = 0.0f;
	*a3 = 0.0f;
	*a4 = 0.0f;
	float t, p;

	%for i in range(grid.Q):
		t = feq->${grid.idx_name[i]} - fi->${grid.idx_name[i]};
		p = t * t / fi->${grid.idx_name[i]};
		t /= fi->${grid.idx_name[i]};
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

${device_func} inline float EstimateAlphaSeries(Dist* fi, Dist* feq) {
	float a1, a2, a3, a4;

	ComputeACoeff(fi, feq, &a1, &a2, &a3, &a4);
	float alpha = ${cex(sym.alpha_series())};
	return alpha;
}

// TODO(mjanusz): The entropy calculation should use a sympy expression.
// Calculates the entropy in the ELBM model.
${device_func} inline float CalculateEntropy(Dist* fi) {
	float ent = 0.0f;

	%for i, w in enumerate(grid.entropic_weights):
		ent += fi->${grid.idx_name[i]} * logf(fi->${grid.idx_name[i]} / (${cex(w)}));
	%endfor

	return ent;
}

// Calculates entropy for the mirror state with a given alpha.
${device_func} inline float CalculateEntropyIneq(Dist* fi, Dist* feq, float alpha) {
	float ent = 0.0f;
	float t;

	%for i, w in enumerate(grid.entropic_weights):
		t = fi->${grid.idx_name[i]} + alpha * (feq->${grid.idx_name[i]} - fi->${grid.idx_name[i]});
		ent += t * logf(t / (${cex(w)}));
	%endfor

	return ent;
}

// Calculates d \Delta entropy / d alpha.
${device_func} inline float CalculateEntropyGrowthDerivative(Dist* fi, Dist* feq, float alpha) {
	float dent = 0.0f;
	float t, neq;

	%for i, w in enumerate(grid.entropic_weights):
		neq = feq->${grid.idx_name[i]} - fi->${grid.idx_name[i]};
		t = fi->${grid.idx_name[i]} + alpha * neq;
		dent += neq * (logf(t / (${cex(w)})) + 1.0f);
	%endfor

	return dent;
}

${device_func} inline float EstimateAlphaFromEntropy(Dist* fi, Dist* feq) {
	float ent = CalculateEntropy(fi);
	float alpha = 2.0f;
	int i = 0;

	// Newton's method to find alpha.
	while (true) {
		float ent_ineq = CalculateEntropyIneq(fi, feq, alpha);
		float ent_increase = ent_ineq - ent;
		if (ent_increase < ${cex(entropy_tolerance)}) {
			break;
		}
		alpha = alpha - ent_increase / CalculateEntropyGrowthDerivative(fi, feq, alpha);
		i++;
		if (i > 10000) {
			die();
		}
	}

	if (alpha < 1.0f) {
		%if check_invalid_values:
			printf("Alpha estimated at: %e\n", alpha);
		%endif
		die();
	}
	return alpha;
}

${device_func} inline bool SmallEquilibriumDeviation(Dist* fi, Dist* feq) {
	return false;
	%if grid is not sym.D3Q15:
		return false;
	%else:
		%for i in range(grid.Q):
			if (fabsf((feq->${grid.idx_name[i]} - fi->${grid.idx_name[i]}) / fi->${grid.idx_name[i]}) > 0.01f) {
				return false;
			}
		%endfor

		return true;
	%endif
}
