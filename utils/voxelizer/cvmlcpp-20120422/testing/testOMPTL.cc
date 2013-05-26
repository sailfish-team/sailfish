/***************************************************************************
 *   Copyright (C) 2007 by BEEKHOF, Fokko                                  *
 *   fpbeekhof@gmail.com                                                   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <iostream>
#include <vector>
#include <boost/random/mersenne_twister.hpp>

#include <omptl/omptl_algorithm>
#include <omptl/omptl_numeric>

boost::mt19937 rng;

void doTest(std::vector<int> &p, std::vector<int> &q)
{
	std::vector<int> s(p.size()), t(q.size());
	const int v1 = rng(), v2 = rng();

	const std::size_t d = std::min(q.size(), std::size_t(3));

	assert(omptl::mismatch(p.begin(), p.end(), q.begin()) ==
		 std::mismatch(p.begin(), p.end(), q.begin()));

	// Numeric
	omptl::adjacent_difference(p.begin(), p.end(), s.begin());
	  std::adjacent_difference(p.begin(), p.end(), t.begin());
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	assert(omptl::inner_product(p.begin(), p.end(), q.begin(), 0) ==
		 std::inner_product(p.begin(), p.end(), q.begin(), 0));

	// Algorithm
	omptl::copy(p.begin(), p.end(), s.begin());
	  std::copy(p.begin(), p.end(), t.begin());
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	assert(omptl::adjacent_find(p.begin(), p.end()) ==
		 std::adjacent_find(p.begin(), p.end()));

	std::copy(p.begin(), p.end(), s.begin());
	std::sort(p.begin(), p.end());
	assert(omptl::binary_search(p.begin(), p.end(), v1) ==
		 std::binary_search(p.begin(), p.end(), v1));

	assert(omptl::count(p.begin(), p.end(), v1) ==
		 std::count(p.begin(), p.end(), v1));

	std::copy(p.begin(), p.end(), s.begin());
	assert(omptl::equal(p.begin(), p.end(), s.begin()) ==
		 std::equal(p.begin(), p.end(), s.begin()));

	assert(omptl::equal_range(p.begin(), p.end(), v1) ==
		 std::equal_range(p.begin(), p.end(), v1));

	omptl::fill(s.begin(), s.end(), v1);
	  std::fill(t.begin(), t.end(), v1);
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	std::copy(p.begin(), p.end(), s.begin());
	std::copy(p.begin(), p.end(), t.begin());
	omptl::fill_n(s.begin(), s.size()/2, v1);
	  std::fill_n(t.begin(), t.size()/2, v1);
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	assert(omptl::find(p.begin(), p.end(), v1) ==
		 std::find(p.begin(), p.end(), v1));

	assert(omptl::find_first_of(p.begin(), p.end(),
				q.begin(), q.begin() + d) ==
		 std::find_first_of(p.begin(), p.end(),
				q.begin(), q.begin() + d) );

	assert(omptl::includes(p.begin(), p.end(),
				q.begin(), q.begin() + d) ==
		 std::includes(p.begin(), p.end(),
				q.begin(), q.begin() + d) );

	assert(omptl::lexicographical_compare(p.begin(), p.end(),
						q.begin(), q.end()) ==
		 std::lexicographical_compare(p.begin(), p.end(),
						q.begin(), q.end()));

	assert(omptl::lower_bound(p.begin(), p.end(), v1) ==
		 std::lower_bound(p.begin(), p.end(), v1));

	assert(omptl::min_element(p.begin(), p.end()) ==
		 std::min_element(p.begin(), p.end()));

	assert(omptl::max_element(p.begin(), p.end()) ==
		 std::max_element(p.begin(), p.end()));

	std::copy(p.begin(), p.end(), s.begin());
	std::copy(p.begin(), p.end(), t.begin());
	omptl::remove(s.begin(), s.end(), v1);
	  std::remove(s.begin(), s.end(), v1);
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	std::copy(p.begin(), p.end(), s.begin());
	std::copy(p.begin(), p.end(), t.begin());
	omptl::replace(s.begin(), s.end(), v1, v2);
	  std::replace(s.begin(), s.end(), v1, v2);
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	omptl::replace_copy(p.begin(), p.end(), s.begin(), v1, v2);
	  std::replace_copy(p.begin(), p.end(), t.begin(), v1, v2);
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	omptl::replace_copy(p.begin(), p.end(), s.begin(), v1, v2);
	  std::replace_copy(p.begin(), p.end(), t.begin(), v1, v2);
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	std::copy(p.begin(), p.end(), s.begin());
	omptl::reverse(s.begin(), s.end());
	assert(omptl::mismatch(s.begin(), s.end(), p.rbegin()) ==
		 std::mismatch(s.begin(), s.end(), p.rbegin()));

	omptl::reverse_copy(p.begin(), p.end(), s.begin());
	  std::reverse_copy(p.begin(), p.end(), t.begin());
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

// 	rotate()

// 	omptl::rotate_copy(p.begin(), p.end(), s.begin());
// 	  std::rotate_copy(p.begin(), p.end(), s.begin());
// 	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
// 		 std::mismatch(s.begin(), s.end(), t.begin()));

	assert(omptl::search(p.begin(), p.end(), q.begin(), q.end()) ==
		 std::search(p.begin(), p.end(), q.begin(), q.end()));

	assert(omptl::search_n(p.begin(), p.end(), v1, v2) ==
		 std::search_n(p.begin(), p.end(), v1, v2));

	// Sort
	std::copy(p.begin(), p.end(), s.begin());
	std::copy(p.begin(), p.end(), t.begin());
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));
	omptl::sort(s.begin(), s.end());
	  std::sort(t.begin(), t.end());
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	std::copy(p.begin(), p.end(), s.begin());
	std::copy(p.begin(), p.end(), t.begin());
	omptl::stable_sort(s.begin(), s.end());
	  std::stable_sort(t.begin(), t.end());
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	std::copy(p.begin(), p.end(), s.begin());
	std::copy(p.begin(), p.end(), t.begin());
	omptl::swap_ranges(s.begin(), s.end(), t.begin()); // Swap
	  std::swap_ranges(t.begin(), t.end(), s.begin()); // Swap back
	assert(omptl::mismatch(p.begin(), p.end(), s.begin()) ==
		 std::mismatch(p.begin(), p.end(), s.begin())); // intact ?
	assert(omptl::mismatch(q.begin(), q.end(), t.begin()) ==
		 std::mismatch(q.begin(), q.end(), t.begin())); // intact ?


	omptl::transform(p.begin(), p.end(), s.begin(), std::negate<int>());
	  std::transform(p.begin(), p.end(), t.begin(), std::negate<int>());
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	omptl::transform(p.begin(), p.end(), q.begin(), s.begin(), std::plus<int>());
	  std::transform(p.begin(), p.end(), q.begin(), t.begin(), std::plus<int>());
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	std::copy(p.begin(), p.end(), s.begin());
	std::copy(p.begin(), p.end(), t.begin());
	omptl::unique(s.begin(), s.end());
	  std::unique(t.begin(), t.end());
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	omptl::unique_copy(p.begin(), p.end(), s.begin());
	  std::unique_copy(p.begin(), p.end(), t.begin());
	assert(omptl::mismatch(s.begin(), s.end(), t.begin()) ==
		 std::mismatch(s.begin(), s.end(), t.begin()));

	omptl::sort(p.begin(), p.end());
	assert(omptl::upper_bound(p.begin(), p.end(), v1) ==
		 std::upper_bound(p.begin(), p.end(), v1));
}

int main()
{
	const int N = 2000;
	std::vector<int> p, q;

	p.reserve(N);
	q.reserve(N);

	for (int i = 0; i < N; ++i)
	{
// std::cout << i << std::endl;
		p.clear();
		q.clear();
		for (int j = 0; j < i; ++j)
		{
			p.push_back(rng());
			q.push_back(rng());
		}
		doTest(p, q);
	}

	return 0;
}
