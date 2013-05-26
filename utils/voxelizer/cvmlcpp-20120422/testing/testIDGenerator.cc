#include <cvmlcpp/base/IDGenerators>
#include <algorithm>
#include <vector>
#include <set>
#include <iostream>

using namespace cvmlcpp;


template <typename T>
void test()
{
	const T N = 9999999u;

	std::vector<T> v;
	KeyGenerator<T> kgen;
	assert(kgen.count() == 0u);
	for (T i = 0u; i < 7u; ++i)
	{
		T k = kgen.generate();
		v.push_back(k);
	}

	// Perfect binary tree depth 3 ?
	std::vector<T> vs(v);
	std::sort(vs.begin(), vs.end());
	assert(v[0] == vs[3]);
	assert(v[1] == vs[1]);
	assert(v[2] == vs[5]);
	assert(v[3] == vs[0]);
	assert(v[4] == vs[4]);
	assert(v[5] == vs[2]);
	assert(v[6] == vs[6]);

	for (T i = 7u; i < N; ++i)
	{
		T k = kgen.generate();
		v.push_back(k);
	}
	assert(kgen.count() == N);

	// Re-generate same set.
	kgen.reset();
	assert(kgen.count() == 0u);
	for (T i = 0u; i < N; ++i)
		assert(kgen() == v[i]);

	GUIDGenerator guidgen;
	assert(guidgen.count() == 0u);
	// Generate N+1u GUIDs, count duplicates...
	std::set<GUIDGenerator::value_type> s;
	for (T i = 0u; i <= N; ++i)
	{
		GUIDGenerator::value_type guid = guidgen.generate();
		assert(s.find(guid) == s.end());
		s.insert(guid);
	}
	assert(guidgen.count() == N+1u);
}

int main()
{
	test<unsigned>();
	test<std::size_t>();
	return 0;
}

