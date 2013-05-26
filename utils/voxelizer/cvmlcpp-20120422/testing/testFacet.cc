#include <volume/Facet>
#include <algorithm>
#include <vector>

int main()
{
	using namespace std;
	using namespace cvmlcpp;

	int N = 16649;
	vector<Facet<size_t, float> > v(N);
	for (int i = 0; i < N; ++i)
	{
		v[i].set(1,2,3);
		v[i].normal() = -1;
	}
	std::sort(v.begin(), v.end());
	//std::cout << v[0].normal() << std::endl;
	return 0;
}
