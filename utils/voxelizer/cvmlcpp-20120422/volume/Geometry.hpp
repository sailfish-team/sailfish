/***************************************************************************
 *   Copyright (C) 2007 by F. P. Beekhof                                   *
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

#include <cassert>
#include <limits>
#include <numeric>
#include <cmath>

#include <cvmlcpp/math/Math>
#include <omptl/omptl_algorithm>

namespace cvmlcpp
{

namespace detail
{

// Adaptor for Rotator3D, for Facet-Normals
// Include dirty hack to avoid const-ness of pairs from maps.
template <typename facet_type,
	typename normal_type = typename facet_type::normal_type,
	typename register_type = typename normal_type::value_type>
class GeoFacetNormRotator
{
	public:
		GeoFacetNormRotator(std::size_t axis, register_type angle) :
			_rotator(axis, angle) { }

		void operator()(const facet_type &f)
		{
			// Write to facet anyway, normal can safely be altered.
			normal_type * const p = const_cast<normal_type *>(&f.normal());
			*p = _rotator(f.normal());
 			//const_cast<facet_type>(f).normal() = _rotator(f.normal());
		}

	private:
		Rotator3D<normal_type> _rotator;
};

// Adaptor for Rotator3D for pair<key, vector>
template <typename Vector, typename register_type = typename Vector::value_type>
class GeoRotator
{
	public:
		GeoRotator(std::size_t axis, register_type angle) :
			_rotator(axis, angle) { }

		const Vector operator()(const Vector &v)
		{ return _rotator(v); }

	private:
		Rotator3D<Vector> _rotator;
};

} // end namespace detail

template <typename T>
Geometry<T>::Geometry()
{
	this->clear();
}

template <typename T>
Geometry<T>::Geometry(const Geometry &that)
{
	*this = that;
}

template <typename T>
template <typename PointIterator, typename NormalIterator>
Geometry<T>::Geometry(const PointIterator begin, const PointIterator end, 
			const NormalIterator normals_begin)
{
	this->loadGeometry(begin, end, normals_begin);
}

template <typename T>
void Geometry<T>::swap(Geometry &that)
{
	_points.swap(that._points);
	_pointKeyMap.swap(that._pointKeyMap);
	_facets.swap(that._facets);
	_facetKeyMap.swap(that._facetKeyMap);
	_pointFacetMap.swap(that._pointFacetMap);
	_pointNormals.swap(that._pointNormals);
	_dirtyFacetNormals.swap(that._dirtyFacetNormals);
	_dirtyPointNormals.swap(that._dirtyPointNormals);
	std::swap(_min, that._min);
	std::swap(_max, that._max);
	std::swap(_minMaxDirty, that._minMaxDirty);
	std::swap( _facetNormalsDirty, that._facetNormalsDirty);
	std::swap(_pointNormalsDirty, that._pointNormalsDirty);
	std::swap(_ptKeygen, that._ptKeygen);
	std::swap(_ftKeygen, that._ftKeygen);
}

template <typename T>
Geometry<T> &Geometry<T>::operator=(const Geometry &that)
{
	this->_ptKeygen = that._ptKeygen;
	this->_ftKeygen = that._ftKeygen;

	if (that._minMaxDirty)
		that.recomputeMinMax();
	for (unsigned i = 0u; i < 3u; ++i)
	{
		this->_min[i] = that.min(i);
		this->_max[i] = that.max(i);
	}

	_points.clear();
	_facets.clear();
	_pointKeyMap.clear();
	_facetKeyMap.clear();
	_dirtyFacetNormals.clear();
	_dirtyPointNormals.clear();

	this->_pointNormals  = that._pointNormals;
	this->_pointFacetMap = that._pointFacetMap;

/*
	std::copy(that._pointNormals.begin(), that._pointNormals.end(),
		  this->_pointNormals.begin());
	std::copy(that._pointFacetMap.begin(), that._pointFacetMap.end(),
		  this->_pointFacetMap.begin());
*/
	if(that._facetNormalsDirty)
	{
		assert(that._pointNormalsDirty);
		that.recomputeFacetNormals();
	}
	if (that._pointNormalsDirty)
		that.recomputePointNormals();

	// Copy points
	for (const_pointkey_iterator pkIt = that._pointKeyMap.begin();
	     pkIt != that._pointKeyMap.end(); ++pkIt)
	{
		// Insert point in local set of points
		std::pair<point_iterator, bool> insertResult =
			_points.insert(*pkIt->second);
		assert(insertResult.second);
		// Store local iterator under that same key as in "that"
		_pointKeyMap[pkIt->first] = insertResult.first;
	}

	// Copy facets
	for (const_facetkey_iterator fkIt = that._facetKeyMap.begin();
	     fkIt != that._facetKeyMap.end(); ++fkIt)
	{
		// Insert facet in local set of facets
		std::pair<facet_iterator, bool> insertResult =
			_facets.insert(*fkIt->second);
		assert(insertResult.second);
		// Store local iterator under that same key as in "that"
		_facetKeyMap[fkIt->first] = insertResult.first;
	}

	_facetNormalsDirty = that._facetNormalsDirty;
	_pointNormalsDirty = that._pointNormalsDirty;
	_minMaxDirty	   = that._minMaxDirty;

	return *this;
}


template <typename T>
template <typename PointIterator, typename NormalIterator>
bool Geometry<T>::loadGeometry(const PointIterator begin, const PointIterator end, const NormalIterator norm_begin)
{
//std::cerr << "points: " << std::distance(begin, end) << " facets: " << (std::distance(begin, end) / 3) << std::endl;
	this->clear();
	if (begin == end)
		return true;

	// Each facet consist of 3 points, so nr of points must be multiple of 3
	if ( std::distance(begin, end) % 3 != 0 )
		return false;

	// Create sorted list of unique points
	std::vector<point_type> uniq_pts(begin, end);
	typedef typename std::vector<point_type>::iterator Pt_It;
	std::sort(uniq_pts.begin(), uniq_pts.end());
	const Pt_It uniq_end = std::unique(uniq_pts.begin(), uniq_pts.end(), roughly_equal_to<point_type>());
	uniq_pts.resize(std::distance(uniq_pts.begin(), uniq_end));
	assert(uniq_pts.end() == uniq_end);
	assert(!uniq_pts.empty());

#ifndef NDEBUG
	for (auto pit = begin; pit != end; ++pit)
		assert(binary_search(uniq_pts.begin(), uniq_pts.end(), *pit, roughly_equal_to<point_type>()));
	//std::cout << "all facet points found" << std::endl;
#endif


	/*
	 * Create balanced tree of points
	 */
	std::vector< std::pair<Pt_It, Pt_It> > ptRange[2];
	unsigned e = 0;
	unsigned o = 1;
	assert(uniq_pts.begin() != uniq_pts.end());
	ptRange[e].push_back(std::make_pair(uniq_pts.begin(), uniq_pts.end()));
	std::vector<std::size_t> pt_keys(uniq_pts.size());
	while (!ptRange[e].empty())
	{
		assert(ptRange[o].empty());
		for (typename std::vector< std::pair<Pt_It, Pt_It> >::const_iterator
		     it = ptRange[e].begin(); it != ptRange[e].end(); ++it)
		{
			const PointIterator begin1 = it->first;
			const PointIterator end2   = it->second;
			assert(begin1 < end2);

			const std::size_t dist = std::distance(begin1, end2);

			const PointIterator middle = begin1 + dist / 2u;
			const std::size_t pt_key = _ptKeygen();

			typedef typename std::set<point_type>::const_iterator PSI;
			const point_type middle_point = *middle;
			const  std::pair<PSI, bool> insert_result = _points.insert(middle_point);
			assert(insert_result.second); // inserted point should be a new element

			typedef typename std::map<std::size_t, PSI>::const_iterator MSI;

			#ifndef NDEBUG
			const std::pair<MSI, bool> map_insert_result =
			#endif
				_pointKeyMap.insert( std::make_pair(pt_key, insert_result.first) );
			assert(map_insert_result.second); // should be a new key

			// Save key for this unique point
			const std::size_t index = std::distance(uniq_pts.begin(), middle);
			assert(index < uniq_pts.size());
			pt_keys[index] = pt_key;

			const PointIterator end1   = middle;
			const PointIterator begin2 = middle + 1;

			if (begin1 != end1)
				ptRange[o].push_back( std::make_pair(begin1, end1) );
			if (begin2 != end2)
				ptRange[o].push_back( std::make_pair(begin2, end2) );
		}
		ptRange[e].clear();
		std::swap(o, e);
	}
	assert(_pointKeyMap.size() == uniq_pts.size());
#ifndef NDEBUG
	// make sure all unique points are accounted for
	for (auto pit = uniq_pts.begin(); pit != uniq_pts.end(); ++pit)
	{
		// find current point
		bool found = false;
		for (auto pkit = _pointKeyMap.begin(); 
		     pkit != _pointKeyMap.end(); ++pkit)
		{
//std::cout << (*(pkit->second)) << " " << (*pit) << std::endl;
			if (*(pkit->second) == *pit)
			{
				found = true;
				break;
			}
		}
		assert(found); // Oopsie ?
	}
	//std::cout << "all unique points found" << std::endl;
#endif

	/*
	 * Gather Facets
	 */
	const std::size_t NFacets = std::distance(begin, end) / 3u;
	std::vector<facet_type> facets( (NFacets) );

	roughly_equal_to<point_type> roughly_equal;

#ifdef _OPENMP
	#pragma omp parallel for
#endif
	for (int f = 0; f < int(NFacets); ++f) // For each facet ...
	{
		for (std::size_t p = 0; p < 3; ++p) // ... there are 3 pts in input
		{
			assert(begin + 3*f+p < end);
			const point_type pt = *(begin + 3*f+p); // index of pt is 3*f+p

			// Find this point of the facet in the vector of unique points
			//const Pt_It pti = std::lower_bound(uniq_pts.begin(), uniq_pts.end(), pt); // doesn't work for roughly_equal
			Pt_It pti = uniq_pts.begin();
			while (!roughly_equal(*pti, pt) )
			{
				++pti;
				assert(pti != uniq_pts.end());
			}
			assert(roughly_equal(*pti, pt));

			// Index in uniq_pts corresponds to index in pt_keys
			std::size_t uniq_index = std::distance(uniq_pts.begin(), pti);
			assert(uniq_index < uniq_pts.size());

			// So the key of the point goes into the facet
			assert(uniq_index < pt_keys.size());
			assert(p < facets[f].size());
			facets[f][p] = pt_keys[uniq_index]; // BUG HERE
		}
		facets[f].normal() = *(norm_begin+f); // Add normal too
	}

	std::sort(facets.begin(), facets.end());
	typedef typename std::vector<facet_type>::iterator FI;
	const FI facets_end =
		std::unique(facets.begin(), facets.end());
	facets.resize( std::distance(facets.begin(), facets_end));

	/*
	 * Create balanced tree of facets
	 */
	assert(_facets.empty());
	assert(_facetKeyMap.empty());
	assert(_pointFacetMap.empty());
	std::vector< std::pair<FI, FI> > fRange[2];
	e = 0;
	o = 1 - e;
	fRange[e].push_back( std::make_pair(facets.begin(), facets.end()) );
	while (!fRange[e].empty())
	{
		assert(fRange[o].empty());
		for (typename std::vector< std::pair<FI, FI> >::const_iterator
		     it = fRange[e].begin(); it != fRange[e].end(); ++it)
		{
			const FI begin1 = it->first;
			const FI end2   = it->second;

			const std::size_t dist = std::distance(begin1, end2);

			const FI middle = begin1 + dist / 2u;

			const facet_type &current_facet = *middle;

			typedef typename std::set<facet_type>:: const_iterator FSCI;
			const std::pair<FSCI, bool> facet_insert_result = _facets.insert( current_facet );
			assert(facet_insert_result.second);

			const std::size_t facet_key = _ftKeygen();
			_facetKeyMap.insert( std::make_pair(facet_key, facet_insert_result.first) );

			for (std::size_t p = 0; p < 3; ++p)
				_pointFacetMap[current_facet[p]].insert(facet_key);

			const FI end1   = middle;
			const FI begin2 = middle + 1;

			if (begin1 != end1)
				fRange[o].push_back( std::make_pair(begin1, end1) );
			if (begin2 != end2)
				fRange[o].push_back( std::make_pair(begin2, end2) );
		}
		fRange[e].clear();
		using std::swap;
		swap(o, e);
	}

	for (facet_iterator fIt = _facets.begin(); fIt != _facets.end(); ++fIt)
		_dirtyPointNormals.insert(fIt->begin(), fIt->end());

	assert(_ftKeygen.count() == this->nrFacets());
	assert(facets.size() == this->nrFacets());

	_minMaxDirty = true;
	this->setNormalsDirty();

	return true;
}


template <typename T>
bool Geometry<T>::operator==(const Geometry &that) const
{
	// quick tests
	if ( (this->nrFacets() != that.nrFacets()) ||
	     (this->nrPoints() != that.nrPoints()) )
		return false;

	for (std::size_t i = 0u; i < 3u; ++i)
		if ( (this->min(i) != that.min(i)) ||
		     (this->max(i) != that.max(i)) )
			return false;

	// Compare points.
	assert(this->_pointKeyMap.size() == that._pointKeyMap.size());
	for (const_pointkey_iterator pkItThis = this->_pointKeyMap.begin(),
	     pkItThat = that. _pointKeyMap.begin();
	     pkItThat != that._pointKeyMap.end(); ++pkItThis, ++pkItThat)
		if ( (pkItThis->first != pkItThat->first) ||  // compare point key
		     (*pkItThis->second != *pkItThat->second) ) // compare point
			return false;

	assert(this->_facetKeyMap.size() == that._facetKeyMap.size());
	for (const_facetkey_iterator fkItThis = this->_facetKeyMap.begin(),
	     fkItThat = that. _facetKeyMap.begin();
	     fkItThat != that._facetKeyMap.end(); ++fkItThis, ++fkItThat)
		if ( ( fkItThis->first  !=  fkItThat->first) ||  // compare facet key
		     (*fkItThis->second != *fkItThat->second) ) // compare facet
			return false;

	return true;
}

template <typename T>
Geometry<T>::~Geometry()
{
// 	this->clear();
}

template <typename T>
void Geometry<T>::clear()
{
	_ptKeygen.reset();
	_ftKeygen.reset();

	std::fill(_min.begin(), _min.end(),  std::numeric_limits<T>::max());
	std::fill(_max.begin(), _max.end(), -std::numeric_limits<T>::max());

	_points.clear();
	_facets.clear();
	_pointKeyMap.clear();
	_facetKeyMap.clear();

	_pointNormals.clear();
	_pointFacetMap.clear();

	_dirtyFacetNormals.clear();
	_dirtyPointNormals.clear();

	_facetNormalsDirty = false;
	_pointNormalsDirty = false;
	_minMaxDirty	   = false;
}

template <typename T>
const typename Geometry<T>::facet_type &Geometry<T>::
facet(const std::size_t key) const
{
	// Internal Consistency check
	assert(_facetKeyMap.size() == _facets.size());

	// User has asked for non-existent key ?
	assert(_facetKeyMap.find(key) != _facetKeyMap.end());

	return *_facetKeyMap[key];
}

template <typename T>
const typename Geometry<T>::vector_type &Geometry<T>::
facetNormal(const std::size_t key) const
{
	if(_facetNormalsDirty)
	{
		assert(_pointNormalsDirty);
		this->recomputeFacetNormals();
	}

	return this->facet(key).normal();
}

template <typename T>
const typename Geometry<T>::vector_type &Geometry<T>::
pointNormal(const std::size_t key) const
{
	// Internal Consistency check
	assert(_pointNormals.size() == _points.size());

	// User has asked for non-existent key ?
	assert(_pointNormals.find(key) != _pointNormals.end());

	if (_pointNormalsDirty)
		this->recomputePointNormals();

	return _pointNormals[key];
}

template <typename T>
const typename Geometry<T>::point_type &Geometry<T>::
point(const std::size_t key) const
{
	// Internal Consistency check

	// Not true: after scale/translation operations, certain points may be
	// "merged" but known under both original keys
	//assert(_pointKeyMap.size() == _points.size());

	// User has asked for non-existent key ?
	assert(_pointKeyMap.find(key) != _pointKeyMap.end());
	return *_pointKeyMap[key];
}

template <typename T>
const std::set<std::size_t> &Geometry<T>::
facetsHavingPoint(const std::size_t key) const
{
	assert(_pointFacetMap.find(key) != _pointFacetMap.end());
	return _pointFacetMap[key];
}

template <typename T>
std::size_t Geometry<T>::addPoint(const value_type &x,
				  const value_type &y,
				  const value_type &z)
{
	return this->addPoint(point_type(x, y, z));
}

template <typename T>
std::size_t Geometry<T>::addPoint(const point_type &point)
{
	assert(point[X] >= value_type(0.0) || point[X] <= value_type(0.0));
	assert(point[Y] >= value_type(0.0) || point[Y] <= value_type(0.0));
	assert(point[Z] >= value_type(0.0) || point[Z] <= value_type(0.0));


	std::size_t key;
	if (_points.empty())
	{
		// It is really a new point.
		key = _ptKeygen.generate();
		assert(_pointKeyMap.find(key) == _pointKeyMap.end());
		_pointKeyMap[key] = _points.insert(point).first;
		std::fill(_pointNormals[key].begin(),
			  _pointNormals[key].end(), value_type(0));
	}
	else
	{
		const const_point_iterator pIt = _points.lower_bound(point);

		// Does this point exist already ?
		roughly_equal_to<point_type> roughly_equals;
		if ( (pIt != _points.end()) && roughly_equals(*pIt, point) ) //(*pIt == point) )
		{
			for (const_pointkey_iterator pkIt= _pointKeyMap.begin();
			     pkIt != _pointKeyMap.end(); ++pkIt)
				if (pkIt->second == pIt)
					return pkIt->first;
			assert(false);
		}

		// It is really a new point.
		key = _ptKeygen.generate();
		assert(_pointKeyMap.find(key) == _pointKeyMap.end());
		_pointKeyMap[key] = _points.insert(pIt, point);
		std::fill(_pointNormals[key].begin(),
			_pointNormals[key].end(), value_type(0));
	}

	// If dirty, there's no point in updating
	if (!_minMaxDirty)
	{
		for (unsigned i = 0; i < 3u; ++i)
			_min[i] = std::min(_min[i], point[i]);
		for (unsigned i = 0; i < 3u; ++i)
			_max[i] = std::max(_max[i], point[i]);
	}

	return key;
}

template <typename T>
bool Geometry<T>::updatePoint(const std::size_t key, const value_type &x,
			      const value_type &y,   const value_type &z)
{
	return this->updatePoint(key, point_type(x, y, z));
}

template <typename T>
bool Geometry<T>::updatePoint(const std::size_t key, const point_type &p)
{
	// Does this point really exist ?
	pointkey_iterator pkIt = _pointKeyMap.find(key);
	if (pkIt == _pointKeyMap.end())
		return false;

	// Erase old point
	_points.erase(pkIt->second);
	_minMaxDirty = true;

	// Insert new point, update keys
	std::pair<point_iterator, bool> insertResult = _points.insert(p);
	if (!insertResult.second)
		return false;
	pkIt->second = insertResult.first;

	// Mark involved normals to be recomputed
	_dirtyPointNormals.insert(key);
	_dirtyFacetNormals.insert(_pointFacetMap[key].begin(),
				  _pointFacetMap[key].end());

	this->setNormalsDirty();

	return true;
}

template <typename T>
bool Geometry<T>::erasePoint(const std::size_t key)
{
	const pointkey_iterator pkIt = _pointKeyMap.find(key);
	if (pkIt == _pointKeyMap.end())
		return false;
	if (_pointFacetMap[key].size() != 0u)
		return false;

	_points.erase(pkIt->second);
	_pointKeyMap.erase(pkIt);
	_pointFacetMap.erase(key);
	_pointNormals.erase(key);
	_dirtyPointNormals.erase(key);

	_minMaxDirty = true;
	this->setNormalsDirty();

	return true;
}

template <typename T>
typename Geometry<T>::const_facet_iterator Geometry<T>::facetsBegin() const
{
	if(_facetNormalsDirty)
		this->recomputeFacetNormals();
	return _facets.begin();
}

template <typename T>
typename Geometry<T>::const_facet_iterator Geometry<T>::facetsEnd() const
{
	return _facets.end();
}

template <typename T>
std::size_t Geometry<T>::addFacet(const std::size_t a,
				const std::size_t b, const std::size_t c)
{
	return this->addFacet(facet_type(a, b, c));
}

template <typename T>
std::size_t Geometry<T>::addFacet(const facet_type &facet)
{
	// User input ok ? I.e., do the points exist ?
	assert(_pointKeyMap.find(facet[A]) != _pointKeyMap.end());
	assert(_pointKeyMap.find(facet[B]) != _pointKeyMap.end());
	assert(_pointKeyMap.find(facet[C]) != _pointKeyMap.end());

	std::size_t key;

	if (_facets.empty())
	{
		key = _ftKeygen();
		assert(_facetKeyMap.find(key) == _facetKeyMap.end());
		_facetKeyMap[key] = _facets.insert(facet).first;
	}
	else
	{
		const facet_iterator fIt = _facets.lower_bound(facet);

		// Does this facet exist already ?
		if ( (fIt != _facets.end()) && (*fIt == facet) )
		{
			for (const_facetkey_iterator fkIt= _facetKeyMap.begin();
			     fkIt != _facetKeyMap.end(); ++fkIt)
				if (fkIt->second == fIt)
					return fkIt->first;
			assert(false);
		}

		// It is really a new facet.
		key = _ftKeygen();
		assert(_facetKeyMap.find(key) == _facetKeyMap.end());
		_facetKeyMap[key] = _facets.insert(fIt, facet);
	}

	_dirtyPointNormals.insert(facet.begin(), facet.end());
	_pointNormalsDirty = true;

	if (facet.normal() == T(0))
	{
		_dirtyFacetNormals.insert(key);
		_facetNormalsDirty = true;
	}

	_pointFacetMap[facet[A]].insert(key);
	_pointFacetMap[facet[B]].insert(key);
	_pointFacetMap[facet[C]].insert(key);

	return key;
}

template <typename T>
bool Geometry<T>::updateFacet(	const std::size_t key, const std::size_t a,
				const std::size_t b, const std::size_t c)
{
	return this->updateFacet(key, facet_type(a, b, c));
}

template <typename T>
bool Geometry<T>::updateFacet(const std::size_t key, const facet_type &facet)
{
	// Does this facet really exist ?
	facetkey_iterator fkIt = _facetKeyMap.find(key);
	if (fkIt == _facetKeyMap.end())
		return false;

	// Do the points in the new facet exist ?
	if ( (_pointKeyMap.find(facet[A]) == _pointKeyMap.end()) ||
	     (_pointKeyMap.find(facet[B]) == _pointKeyMap.end()) ||
	     (_pointKeyMap.find(facet[C]) == _pointKeyMap.end()) )
		return false;

	// Erase old facet
	_pointFacetMap[(*fkIt->second)[A]].erase(key);
	_pointFacetMap[(*fkIt->second)[B]].erase(key);
	_pointFacetMap[(*fkIt->second)[C]].erase(key);
	_facets.erase(fkIt->second);

	// Insert new facet, update keys
	std::pair<facet_iterator, bool> insertResult = _facets.insert(facet);
	if (!insertResult.second)
		return false;
	fkIt->second = insertResult.first;

	// Mark involved normals to be recomputed
	_dirtyPointNormals.insert(fkIt->second->begin(), fkIt->second->end());
	_dirtyFacetNormals.insert(key);

	_minMaxDirty = true;
	this->setNormalsDirty();

	return true;
}

template <typename T>
bool Geometry<T>::eraseFacet(const std::size_t key)
{
	const facetkey_iterator fkIt = _facetKeyMap.find(key);
	if (fkIt == _facetKeyMap.end())
		return false;

	// Remove from mapping for each point
	const facet_type facet = *fkIt->second;
	_pointFacetMap[facet[0]].erase(key);
	_pointFacetMap[facet[1]].erase(key);
	_pointFacetMap[facet[2]].erase(key);

	_dirtyFacetNormals.erase(key);
	_facets.erase(fkIt->second);
	_facetKeyMap.erase(fkIt);

	this->setNormalsDirty();

	return true;
}

template <typename T>
T Geometry<T>::min(const unsigned dim) const
{
	// Member function should not be called without point
	assert(this->nrPoints() > 0);

	if (_minMaxDirty)
		this->recomputeMinMax();

	return _min[dim];
}

template <typename T>
T Geometry<T>::max(const unsigned dim) const
{
	// Member function should not be called without point
	assert(this->nrPoints() > 0);

	if (_minMaxDirty)
		this->recomputeMinMax();

	return _max[dim];
}

template <typename T>
void Geometry<T>::center()
{
	const T dx = (this->max(X) + this->min(X)) * 0.5;
	const T dy = (this->max(Y) + this->min(Y)) * 0.5;
	const T dz = (this->max(Z) + this->min(Z)) * 0.5;

	this->translate(-dx, -dy, -dz);
}

template <typename T>
void Geometry<T>::centerMass()
{
	typename Geometry<T>::point_type mass =
		std::accumulate(this->pointsBegin(),
		 	        this->pointsEnd(), point_type(0.0));

	this->translate(mass * T(-1));
}

template <typename T>
void Geometry<T>::scale(const T factor)
{
	if (this->nrPoints() == 0)
		return;

	// Bogus user input ?
	assert(factor > 0.0);
/*
	// Very dirty, but should be safe, provided factor > 0
	for (const_point_iterator it = this->pointsBegin();
	     it != this->pointsEnd(); ++it)
	{
		point_type * p = const_cast<point_type *>(&(*it));
		*p = *it * factor;
	}
*/
	typedef typename std::map<std::size_t, point_iterator>::const_iterator PKMap_It;
	typedef typename std::set<point_type>::const_iterator PSet_It;
	std::set<point_type> new_points;
	std::map<std::size_t, point_iterator> new_pointKeyMap;

	std::vector<point_iterator> pt_its;
	for (PKMap_It pk_it = _pointKeyMap.begin(); pk_it != _pointKeyMap.end(); ++pk_it)
	{
		const point_iterator pt_it = pk_it->second;
		assert( std::find(pt_its.begin(), pt_its.end(), pt_it) == pt_its.end() );
		pt_its.push_back(pt_it);
	}

	KeyGenerator<std::size_t> reGen;
	for (std::size_t i = 0u; i < _ptKeygen.count(); ++i)
	{
		const std::size_t key = reGen.generate();

		const PKMap_It pkit = _pointKeyMap.find(key);

		if (pkit != _pointKeyMap.end())
		{
			assert(pkit->first == key);
			assert(new_pointKeyMap.find(key) == new_pointKeyMap.end());
			assert(_pointKeyMap.find(key) != _pointKeyMap.end());

			const point_iterator pt_it = pkit->second;
			const point_type point = *pt_it;
			assert(_pointKeyMap.erase(key) == 1);
			assert(_points.erase(point) == 1);
			const point_type new_point = point * factor;
/*
			PSet_It new_pt_it = new_points.find(new_point);
			if (new_pt_it != new_points.end())
			{
				std::cout << "Found " << new_point << " was "<< factor << " * " << point << std::endl;

				//PKMap_It it = new_pointKeyMap.find(new_pt_it);
				for (PKMap_It it = new_pointKeyMap.begin(); it != new_pointKeyMap.end(); it++)
				{
					//assert(it != new_pointKeyMap.end());
					//assert(it->second == new_pt_it);
					if (it->second == new_pt_it)
					{
						std::size_t old_key = it->first;
						const point_type old_point = *_pointKeyMap[old_key];
						std::cout << "Other point was " << old_point << std::endl;
					}
				}
			}
*/
			//assert(new_points.find(new_point) == new_points.end());
			const std::pair<const_point_iterator, bool>
					insResult = new_points.insert(new_point);
			//assert(insResult.second); // Points should be unique
			new_pointKeyMap.insert(std::make_pair(key, insResult.first));
		}
	}
	_points.swap(new_points);
	_pointKeyMap.swap(new_pointKeyMap);
	//assert(_points.size() == _pointKeyMap.size());

/*
	std::map<std::size_t, point_type> tempPts;

	// Scale points and store in temporary map
	for (const_pointkey_iterator it = _pointKeyMap.begin();
	     it != _pointKeyMap.end(); ++it)
		tempPts[it->first] = *(it->second) * factor;

	// Delete old set of points
	_points.clear();

	// Insert new points and update keymap
	typedef typename std::map<std::size_t, point_type>::const_iterator TmpIt;
	for (TmpIt it = tempPts.begin(); it != tempPts.end(); ++it)
	{
		// Insert
		std::pair<const_point_iterator, bool>
			insResult = _points.insert(it->second);

		// Should be impossible
		assert(insResult.second);

		// Update keymap
		_pointKeyMap[it->first] = insResult.first;
	}
*/
	// Fix min & max
	if (!_minMaxDirty)
	{
		_min *= factor;
		_max *= factor;
	}
}

template <typename T>
void Geometry<T>::scaleTo(const T maxLength)
{
	const T dx = this->max(X) - this->min(X);
	const T dy = this->max(Y) - this->min(Y);
	const T dz = this->max(Z) - this->min(Z);

	const T maxLen = std::max( dx, std::max(dy, dz) );
	this->scale(maxLength / maxLen);
}

template <typename T>
template <class Vector>
void Geometry<T>::translate(const Vector &v)
{
	if (this->nrPoints() == 0)
		return;

	// Works only for floating-point types!
	const T e = T(2.0) * std::numeric_limits<T>::min();
	if (!( (std::abs(v[X])>e) || (std::abs(v[Y])>e) || (std::abs(v[Z])>e) ))
		return;

	std::set<point_type> new_points;
	std::map<std::size_t, point_iterator> new_pointKeyMap;
/*
	// Useless test of uniqueness. Points can be known under different
	// keys after "merging" due to numerical issues.
	std::vector<point_iterator> pt_its;
	for (typename std::map<std::size_t, point_iterator>::const_iterator
	     pkit = _pointKeyMap.begin(); pkit != _pointKeyMap.end(); ++pkit)
	{
		const point_iterator pt_it = pkit->second;
		assert( std::find(pt_its.begin(), pt_its.end(), pt_it) == pt_its.end() );
		pt_its.push_back(pt_it);
	}
*/
	KeyGenerator<std::size_t> reGen;
	for (std::size_t i = 0u; i < _ptKeygen.count(); ++i)
	{
		const std::size_t key = reGen.generate();

		const typename std::map<std::size_t, point_iterator>::const_iterator
			pkit = _pointKeyMap.find(key);

		if (pkit != _pointKeyMap.end())
		{
			assert(pkit->first == key);
			assert(new_pointKeyMap.find(key) == new_pointKeyMap.end());
			assert(_pointKeyMap.find(key) != _pointKeyMap.end());

			const point_iterator pt_it = pkit->second;
			const point_type point = *pt_it;
			//assert(_pointKeyMap.erase(key) == 1);
			//assert(_points.erase(point) == 1);
			const point_type new_point = point + v;

/*			if (new_points.find(new_point) != new_points.end())
				std::cout << "Found " << new_point << " was "<< point << " + " << v << std::endl;

			PSet_It new_pt_it = new_points.find(new_point);
			if (new_pt_it != new_points.end())
			{
				std::cout << "Found " << new_point << " was "<< factor << " * " << point << std::endl;

				//PKMap_It it = new_pointKeyMap.find(new_pt_it);
				for (PKMap_It it = new_pointKeyMap.begin(); it != new_pointKeyMap.end(); it++)
				{
					//assert(it != new_pointKeyMap.end());
					//assert(it->second == new_pt_it);
					if (it->second == new_pt_it)
					{
						std::size_t old_key = it->first;
						const point_type old_point = *_pointKeyMap[old_key];
						std::cout << "Other point was " << old_point << std::endl;
					}
				}
			}
*/
			//assert(new_points.find(new_point) == new_points.end());
			const std::pair<const_point_iterator, bool>
					insResult = new_points.insert(new_point);
			//assert(insResult.second); // Points should be unique
			new_pointKeyMap.insert(std::make_pair(key, insResult.first));
		}
	}
	_points.swap(new_points);
	_pointKeyMap.swap(new_pointKeyMap);
	//assert(_points.size() == _pointKeyMap.size());

/*	typedef std::pair<std::size_t, point_type> pPair;
	std::vector<pPair> tempPts;

	// Translate points and store in temporary map
	for (const_pointkey_iterator it = _pointKeyMap.begin();
	     it != _pointKeyMap.end(); ++it)
	{
		tempPts.push_back(std::make_pair(it->first, *it->second + v));
	}

	// Delete old set of points
	assert(_ptKeygen.count() >= this->nrPoints());
	_points.clear();
	_pointKeyMap.clear();

	// Insert new points and update keymap
	typedef typename std::vector<pPair>::const_iterator TmpPIt;
	typedef PairFirstCompare<std::size_t, point_type,
			std::less<std::size_t> > PairLess;

	KeyGenerator<std::size_t> reGen;
	for (std::size_t i = 0u; i < _ptKeygen.count(); ++i)
	{
		const std::size_t key = reGen.generate();
		const TmpPIt it = std::lower_bound(tempPts.begin(),
						tempPts.end(), key, PairLess());

		if ( (it != tempPts.end()) && (it->first == key) )
		{
			// Insert
			std::pair<const_point_iterator, bool>
				insResult = _points.insert(it->second);
			//assert(insResult.second); // Should be impossible

			// Update keymap
			_pointKeyMap.insert(std::make_pair(key, insResult.first));
		}
	}
*/	assert(reGen.count() == _ptKeygen.count());

	// Fix min & max
	if (!_minMaxDirty)
	{
		_min += v;
		_max += v;
	}
}

template <typename T>
void Geometry<T>::translate(const T dx, const T dy, const T dz)
{
	this->translate(vector_type(dx, dy, dz));
}

template <typename T>
void Geometry<T>::rotate(const std::size_t axis, const T angle)
{
	// Used supplied invalid axis ?
	assert( (axis == X) || (axis == Y) || (axis == Z) );

	if (this->nrPoints() == 0)
		return;

	// Works only for floating-point types!
	const T e = T(2.0) * std::numeric_limits<T>::min();
	if ( std::abs(std::fmod(angle, T(2.0)*Constants<T>::pi())) < e )
		return;

	/*
	 * Rotate Points
	 */

	// Rotate and place in temporary map
	typedef std::pair<std::size_t, point_type> pPair;
	std::vector<pPair> tempKeyPts;
	tempKeyPts.reserve(_pointKeyMap.size());
//	std::vector<point_type> tempPts;
//	tempPts.reserve(_pointKeyMap.size());

	detail::GeoRotator<point_type> grp((axis), angle);

	for (const_pointkey_iterator it = _pointKeyMap.begin();
	     it != _pointKeyMap.end(); ++it)
	{
		const point_type new_pt = grp(*(it->second));
		tempKeyPts.push_back( std::make_pair(it->first, new_pt) );
//		tempPts.push_back(new_pt);
	}

	typedef PairFirstCompare<std::size_t, point_type,
					std::less<std::size_t> > PairLess;
	std::sort(tempKeyPts.begin(), tempKeyPts.end(), PairLess());
//	std::sort(tempPts.begin(), tempPts.end());
//	std::unique(tempPts.begin(), tempPts.end(), roughly_equal<point_type>());

	// Insert new points 
/*	KeyGenerator<std::size_t> reGen;
	_points.clear();
	std::vector< std::pair<Pt_It, Pt_It> > ptRange[2];
	unsigned e = 0;
	unsigned o = 1;
	ptRange[e].push_back(std::make_pair(uniq_pts.begin(), uniq_pts.end()));
	std::vector<std::size_t> pt_keys(uniq_pts.size());
	while (!ptRange[e].empty())
	{
		assert(ptRange[o].empty());
		for (typename std::vector< std::pair<Pt_It, Pt_It> >::const_iterator
		     it = ptRange[e].begin(); it != ptRange[e].end(); ++it)
		{
			const PointIterator begin1 = it->first;
			const PointIterator end2   = it->second;

			const std::size_t dist = std::distance(begin1, end2);

			const PointIterator middle = begin1 + dist / 2u;
			const std::size_t pt_key = reGen.generate();

			typedef typename std::set<point_type>::const_iterator PSI;
			const  std::pair<PSI, bool> insert_result = _points.insert(*middle);
			assert(insert_result.second); // inserted point should be a new element

			typedef typename std::map<std::size_t, PSI>::const_iterator MSI;

			#ifndef NDEBUG
			const std::pair<MSI, bool> map_insert_result =
			#endif
				_pointKeyMap.insert( std::make_pair(pt_key, insert_result.first) );
			assert(map_insert_result.second); // should be a new key

			// Save key for this unique point
			const std::size_t index = std::distance(uniq_pts.begin(), middle);
			assert(index < uniq_pts.size());
			pt_keys[index] = pt_key;

			const PointIterator end1   = middle;
			const PointIterator begin2 = middle + 1;

			if (begin1 != end1)
				ptRange[o].push_back( std::make_pair(begin1, end1) );
			if (begin2 != end2)
				ptRange[o].push_back( std::make_pair(begin2, end2) );
		}
		ptRange[e].clear();
		std::swap(o, e);
	}
*/
	
	//  update keymap
	KeyGenerator<std::size_t> reGen;
	_points.clear();
	_pointKeyMap.clear();
	for (std::size_t i = 0u; i < _ptKeygen.count(); ++i)
	{
		typedef typename std::vector<pPair>::const_iterator TmpPIt;

		const std::size_t key = reGen.generate();
		const TmpPIt it = std::lower_bound(tempKeyPts.begin(),
						tempKeyPts.end(), key, PairLess());
		if ( (it != tempKeyPts.end()) && (it->first == key) )
		{
			// Insert
			std::pair<const_point_iterator, bool>
				insResult = _points.insert(it->second);
			assert(insResult.second); // Should be impossible

			// Update keymap
			_pointKeyMap.insert(std::pair<std::size_t, point_iterator>
						(it->first, insResult.first));
		}
	}

	/*
	 * Rotate Normals
	 */
	std::for_each(_facets.begin(), _facets.end(),
			detail::GeoFacetNormRotator<facet_type>(axis, angle));

	/*
	 * Rotate Point-Normals
	 */
	std::map<std::size_t, vector_type> newPtNormals;
	typedef MapPairOperateInserter<
				std::map<std::size_t, vector_type>,
				detail::GeoRotator<vector_type> > PointNormsInserter;


	detail::GeoRotator<vector_type> grv(axis, angle);
	PointNormsInserter pni((newPtNormals), grv);
	std::for_each(_pointNormals.begin(), _pointNormals.end(), pni);

	_pointNormals.swap(newPtNormals);

	// Didn't fix min/max
	_minMaxDirty = true;
}

template <typename T>
void Geometry<T>::recomputeMinMax() const
{
	std::fill(_min.begin(), _min.end(),  std::numeric_limits<T>::max());
	std::fill(_max.begin(), _max.end(), -std::numeric_limits<T>::max());

	for (typename std::set<point_type>::const_iterator p = _points.begin();
	     p != _points.end(); ++p)
	{
		for (std::size_t i = 0; i < 3u; ++i)
			_min[i] = std::min(_min[i], (*p)[i]);
		for (std::size_t i = 0; i < 3u; ++i)
			_max[i] = std::max(_max[i], (*p)[i]);
	}

	_minMaxDirty = false;
}

template <typename T>
void Geometry<T>::recomputeFacetNormals() const
{
	// Recompute facet normals
	for (std::set<std::size_t>::const_iterator
	     it = _dirtyFacetNormals.begin();
	     it != _dirtyFacetNormals.end(); ++it)
	{
// std::cout << "recomputeFacetNormals() Facet: " << *it << std::endl;
		facet_type fac = *_facetKeyMap[*it];

		// Valid Facet ?
		assert(fac[A] != fac[B]);
		assert(fac[A] != fac[C]);
		assert(fac[B] != fac[C]);
// std::cout << "recomputeFacetNormals() abc: " << fac[A] << " " << fac[B]
// 	<< " " << fac[C] << std::endl;
		// Points exist ?
		assert(_pointKeyMap.find(fac[A]) != _pointKeyMap.end());
		assert(_pointKeyMap.find(fac[B]) != _pointKeyMap.end());
		assert(_pointKeyMap.find(fac[C]) != _pointKeyMap.end());

		// Changing the facet normal will change the point normals
		_dirtyPointNormals.insert(fac[A]);
		_dirtyPointNormals.insert(fac[B]);
		_dirtyPointNormals.insert(fac[C]);

		const vector_type ab =
				*_pointKeyMap[fac[B]] - *_pointKeyMap[fac[A]];
		const vector_type ac =
				*_pointKeyMap[fac[C]] - *_pointKeyMap[fac[A]];
		assert(dotProduct(ab, ab) > 0.0f);
		assert(dotProduct(ac, ac) > 0.0f);

		const vector_type norm = crossProduct(ac, ab);
		if (!(dotProduct(norm, norm) > 0.0f))
		{
			// Computation of normal failed! Bugger. Now what ?
			facet_type *p = const_cast<facet_type *>
						(&(*_facetKeyMap[*it]));
			p->normal() = vector_type(value_type(0.0));
			continue;
		}
		assert(dotProduct(norm, norm) > 0.0f);
		facet_type *p = const_cast<facet_type *>(&(*_facetKeyMap[*it]));
		p->normal() = norm / std::sqrt(dotProduct(norm, norm));
	}

	_dirtyFacetNormals.clear();
	_facetNormalsDirty = false;
}

template <typename T>
void Geometry<T>::recomputePointNormals() const
{
	if(_facetNormalsDirty)
		this->recomputeFacetNormals();

	// Recompute point normals
	for (std::set<std::size_t>::const_iterator
	     it = _dirtyPointNormals.begin();
	     it != _dirtyPointNormals.end(); ++it)
	{
		vector_type pNormal(value_type(0.0));

		for (std::set<std::size_t>::const_iterator
		     normIt = _pointFacetMap[*it].begin();
		     normIt != _pointFacetMap[*it].end(); ++normIt)
			pNormal += _facetKeyMap[*normIt]->normal();
// 		pNormal /= value_type(_pointFacetMap[*it].size());
		pNormal /= std::sqrt(dotProduct(pNormal, pNormal));

		_pointNormals[*it] = pNormal;
	}

	_dirtyPointNormals.clear();
	_pointNormalsDirty = false;
}

template <typename T>
inline void Geometry<T>::setNormalsDirty()
{
	_facetNormalsDirty = true;
	_pointNormalsDirty = true;
}

} // namespace cvmlcpp
