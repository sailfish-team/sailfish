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

// #include <iostream>

#include <set>
#include <vector>

#include <algorithm>
#include <numeric>
#include <functional>
#include <cassert>

#include <tr1/memory>
#include <tr1/array>

#include <cvmlcpp/base/Allocators>

#include <cvmlcpp/base/use_omp.h>
#include <omptl/omptl_algorithm>

#ifdef __GXX_EXPERIMENTAL_CXX0X__
#include <initializer_list>
#endif

namespace cvmlcpp
{

template <typename T, std::size_t D, class DataHolder>
class _ConstRefMatrix;

template <typename T, std::size_t D, class DataHolder>
class _RefMatrix;

template <typename T>
struct ValueRange
{
	typedef T value_type;

	ValueRange(const std::size_t begin = 0u, const std::size_t end = 0u,
		   const T value = T(0) ) :
		_begin(begin), _end(end), _value(value)
	{ }

	bool operator< (const ValueRange &that) const
	{ return this->_begin < that._begin; }

	bool operator< (const std::size_t that) const
	{ return this->_begin < that; }

	std::size_t _begin, _end;
	T _value;
};

template <class CDH>
class _CDHIterator
{
	public:
		typedef std::ptrdiff_t			difference_type;
		typedef std::random_access_iterator_tag iterator_category;

		_CDHIterator(CDH &cdh, const std::size_t index = 0u) :
			_cdh(cdh), _index(index) { }

		const typename CDH::value_type& operator*() const
		{
// std::cout << "Iterator: index " << _index << std::endl;
			return _cdh[_index];
		}

		typename CDH::value_type& operator*()
		{
// std::cout << "Iterator: index " << _index << std::endl;
			return _cdh[_index];
		}

		typename CDH::value_type* const operator->()
		{
			return &_cdh[_index];
		}

		const _CDHIterator &operator++()
		{
			++_index;
			return *this;
		}

		_CDHIterator operator+(const std::size_t i) const
		{ return _CDHIterator(this->_cdh, this->_index + i); }

		const _CDHIterator &operator+=(const std::size_t i)
		{ _index += i; return *this; }

	private:
		CDH &_cdh;
		std::size_t _index;
};


template <class CDH>
class _CDHConstIterator
{
	public:
		typedef std::ptrdiff_t			difference_type;
		typedef std::random_access_iterator_tag iterator_category;

		_CDHConstIterator(CDH &cdh, const std::size_t index = 0u) :
			_cdh(cdh), _index(index) { }

		const typename CDH::value_type operator*() const
		{
// std::cout << "ConstIterator: index " << _index << std::endl;
			return _cdh[_index];
		}

		const typename CDH::value_type* const operator->() const
		{
// std::cout << "ConstIterator: index " << _index << std::endl;
			return &_cdh[_index];
		}

		const _CDHConstIterator &operator++()
		{
			++_index;
			return *this;
		}

		_CDHConstIterator operator+(const std::size_t i) const
		{ return _CDHConstIterator(this->_cdh, this->_index + i); }

	private:
		/*mutable*/ CDH &_cdh;
		std::size_t _index;
};

template <class CDH, class OtherIterator>
bool const operator==(const _CDHIterator<CDH> &lhs, const OtherIterator &rhs)
{
	return (&(*lhs) == &(*rhs));
}

template <class CDH, class OtherIterator>
bool const operator!=(const _CDHIterator<CDH> &lhs, const OtherIterator &rhs)
{
	return !(lhs == rhs);
}

template <typename T>
class CompressDataHolder
{
	private:
		typedef typename std::set<ValueRange<T> >::iterator VRIterator;
		typedef typename std::set<ValueRange<T> >::const_iterator
								CVRIterator;
		typedef std::pair<VRIterator, bool> InsertResult;

	public:
		typedef T value_type;
		typedef T* pointer;
		typedef T& reference;
		typedef const T& const_reference;
		typedef _CDHIterator<CompressDataHolder<T> > iterator;
		typedef _CDHConstIterator<CompressDataHolder<T> >const_iterator;

		CompressDataHolder(const std::size_t capacity = 0u) :
			_capacity(capacity), _nullValue(T(0))
		{
			InsertResult insResult = _data.insert(
				ValueRange<T>(0u, _capacity, _nullValue));
			assert(insResult.second);
		}

		CompressDataHolder(const std::size_t capacity,
				   const T &value) :
					_capacity(capacity), _nullValue(T(0))
		{
			InsertResult insResult = _data.insert(
				ValueRange<T>(0u, _capacity, value));
			assert(insResult.second);
		}

		const_reference operator[](const std::size_t index) const
		{
// std::cout << "CDH const[] index: " << index << std::endl;

			// A dirty hack to support (operator== called with end()
			if (index == _capacity)
			{
// std::cout << "NULL" << std::endl;
				return _nullValue;
			}

			assert(index < _capacity);
   			this->cleanup();

			CVRIterator ptr = _data.lower_bound(index);

			if (ptr != _data.end())
			{
				if ( (ptr != _data.begin()) &&
				     (ptr->_begin > index) )
					--ptr;

				if ( (ptr->_begin <= index) &&
				     (index < ptr->_end) )
					return ptr->_value;
			}

			// You are reading uninitialized data!
			assert(false);
			return _nullValue;
		}

		reference operator[](const std::size_t index)
		{
			// A dirty hack to support (operator== called with end()
			if (index == _capacity)
				return *const_cast<T*>(&_nullValue);

// std::cout << "CDH [] index: " << index << std::endl;

			assert(index < _capacity);

			this->cleanup();

			if (_data.empty())
			{
				InsertResult insResult =
				    _data.insert(ValueRange<T>(index, index+1));
				assert(insResult.second);
				return *const_cast<T*>
						(&(insResult.first->_value));
			}

			VRIterator ptr = _data.lower_bound(index);

			if ( (ptr == _data.end()) ||
			     ( (ptr != _data.begin()) &&
				     (ptr->_begin > index) ) )
					--ptr;

// std::cout << "CDH [] lower_bound :(" <<
// ptr->_begin <<", " << ptr->_end << ", " << ptr->_value << ") " << std::endl;

			// Uglyness to allow us to modify the "_value"
			// part of this range, which is not used in
			// operator<, and hence is safe.
			if (ptr->_begin == index)
			{
				if (ptr->_end == index+1u)
				{
// std::cout << "CDH [] FOUND: (" <<
// ptr->_begin <<", " << ptr->_end << ", " << ptr->_value << ") " << std::endl;

					return *const_cast<T*>(&(ptr->_value));
				}
				else
				{
/*std::cout << "CDH [] MOVE: (" <<
ptr->_begin <<", " << ptr->_end << ", " << ptr->_value << ") " << std::endl;*/
					assert(ptr->_end > index+1u);
					const T value = ptr->_value;
					const std::size_t end = ptr->_end;

					_data.erase(ptr);

					InsertResult insRes = _data.insert(
					    ValueRange<T>(index+1u, end,value));
					assert(insRes.second);
					insRes = _data.insert(
					  ValueRange<T>(index, index+1u,value));
					assert(insRes.second);
					return *const_cast<T*>
						(&(insRes.first->_value));
				}
			}

			assert(ptr->_begin < index);
			InsertResult insRes;
			if (ptr->_end > index+1u)
			{
				assert(ptr->_end > index+1u);
// std::cout << "CDH [] SPLIT: (" <<
// ptr->_begin <<", " << ptr->_end << ", " << ptr->_value << ") " << std::endl;

				insRes = _data.insert(ValueRange<T>
					    (index+1u, ptr->_end, ptr->_value));
				assert(insRes.second);
				*const_cast<std::size_t *>(&(ptr->_end)) = index;
				insRes = _data.insert(
				  ValueRange<T>(index, index+1u, ptr->_value));
			}
			else if (ptr->_end == index+1u)
			{
// std::cout << "CDH [] TRUNC: (" <<
// ptr->_begin <<", " << ptr->_end << ", " << ptr->_value << ") " << std::endl;

				*const_cast<std::size_t *>(&(ptr->_end)) = index;
				insRes = _data.insert(
				  ValueRange<T>(index, index+1u, ptr->_value));
			}
			else
			{
				assert(ptr->_end <= index);
				insRes = _data.insert(
						ValueRange<T>(index, index+1u));
			}
			assert(insRes.second);

			// Uglyness to allow us to modify the "_value" part
			// of this range, which is not used in operator<,
			// and hence is safe.
			return *const_cast<T*>(&(insRes.first->_value));
		}

		void resize(const std::size_t capacity)
		{ this->clear(); _capacity = capacity; }

		std::size_t size() const { return _capacity; }

		void clear() { _data.clear(); }

		iterator begin()
		{ this->cleanup();  return iterator(*this); }

		iterator end() { return iterator(*this, this->size()); }

		const_iterator begin() const
		{ this->cleanup(); return const_iterator(*this); }

		const_iterator end() const
		{ return const_iterator(*this, this->size()); }

		void swap(CompressDataHolder &c)
		{
			std::swap(this->_capacity, c._capacity);
			this->_data.swap(c._data);
		}

	private:
		void cleanup() const
		{
			bool changed = false;

			typedef typename
				std::set<ValueRange<T> >::iterator VRIterator;
bool first = true;
while (first || changed)
{
	first = false;
	changed = false;
for (VRIterator next, it = _data.begin(); it != _data.end(); it = next)
{
	next = it; ++next;

	// Merge with previous
// std::cout << "Cleanup: Inspecting ("
//  << it->_begin <<", " << it->_end << ", " << it->_value << ") " << std::endl;

	assert(it != _data.end());
	if (it != _data.begin())
	{
		VRIterator prev = it; --prev;

// std::cout << "Cleanup: ... prev " <<
//  "(" << prev->_begin <<", " << prev->_end << ", " << prev->_value << ") "
// << std::endl;

		if (prev->_value == it->_value)
		{
			if (prev->_end >= it->_begin)
			{
// std::cout << "Cleanup: ... Merge " << std::endl;
				// Merge
				*const_cast<std::size_t *>(& prev->_end) =
					std::max(it->_end, prev->_end);
				_data.erase(it);
				it = prev;
				changed = true;
			}
		}
		else
		{
			if(prev->_end == it->_end)
			{
// std::cout << "Cleanup: ... Trunc " << std::endl;
				// Truncate
				--(*const_cast<std::size_t *>(& prev->_end) =
					std::max(it->_end, prev->_end));
				changed = true;
			}
			else if(prev->_end > it->_end)
			{
// std::cout << "Cleanup: ... Split " << std::endl;
				// Split
				_data.insert(ValueRange<T>(it->_end,
						prev->_end, prev->_value));
				*const_cast<std::size_t *>(& prev->_end) =
					it->_begin;
				changed = true;
			}
		}
	}

	// Consider next
	if (next != _data.end())
	{
		assert(next->_begin > it->_begin);
// std::cout << "Cleanup: ... next " <<
// "(" << it->_begin <<", " << it->_end << ", " << it->_value << ") --> " <<
// "(" << next->_begin <<", " << next->_end << ", " << next->_value << ") "
// << std::endl;

		if (it->_value == next->_value)
		{
			if (it->_end == next->_begin)
			{
// std::cout << "Cleanup: ... Merge " << std::endl;
				// Merge
				*const_cast<std::size_t *>(&it->_end) = next->_end;
				_data.erase(next);
				next = it; ++next;
				changed = true;
			}
		}
		else
		{
			if(it->_end >= next->_end)
			{
// std::cout << "Cleanup: ... Delete " << std::endl;
				// Delete Next
				_data.erase(next);
				next = it; ++next;
				changed = true;
			}
			else if ( (it->_end > next->_begin) &&
				  (it->_end < next->_end) )
			{
// std::cout << "Cleanup: ... Move " << std::endl;
				// Move
				_data.insert(ValueRange<T>(it->_end,
						next->_end, next->_value));
				_data.erase(next);
				next = it; ++next;
				changed = true;
				if (next != _data.end())
				{
					assert(it->_begin < next->_begin);
					assert(it->_end < next->_begin);
				}
			}
			else if ( (it->_end > next->_begin) &&
				  (it->_end > next->_end) )
			{
// std::cout << "Cleanup: ... Split " << std::endl;
				// Split
				_data.insert(ValueRange<T>(next->_end,
						it->_end, it->_value));
				*const_cast<std::size_t *>(&it->_end) =
					next->_begin;
			}
		}
	}

}
	} // while

// std::cout << "\tCleanup: Exit ";
// for (typename std::set<ValueRange<T> >::iterator it = _data.begin();
// 	it != _data.end(); ++it)
// std::cout << "(" << it->_begin <<", " << it->_end << ", " << it->_value<<")";
// std::cout << std::endl;

for (VRIterator next, it = _data.begin(); it != _data.end(); it = next)
{
	next = it; ++next;
	if (next == _data.end())
		break;
	assert(it->_begin < next->_begin);
	assert(it->_end <= next->_begin);
}

		}

		std::size_t _capacity;
		mutable std::set<ValueRange<T> > _data;
		const T _nullValue;
};

template <typename T, std::size_t D, class DataHolder>
class _RefMatrix
{
	public:
		typedef T value_type;
		typedef typename DataHolder::iterator iterator;
		typedef typename DataHolder::const_iterator const_iterator;
		typedef typename DataHolder::pointer pointer;
		typedef typename DataHolder::reference reference;
		typedef typename DataHolder::const_reference const_reference;

		typedef _RefMatrix<T, D-1u, DataHolder> slice_type;

		_RefMatrix(const Matrix<T, D, DataHolder> &m) :
			_extents(m.extents()), _data(m._data), _offset(0) { }

		_RefMatrix(const typename
			std::tr1::array<std::size_t, D>::const_iterator &extents,
			const typename std::tr1::array<std::size_t, D>::const_iterator &dimsMult,
			  /*const*/ DataHolder &data, const std::size_t offset) :
				_extents(extents), _dimsMult(dimsMult),
				_data(data), _offset(offset)
		{
			assert(_offset < _data.size());
		}

		slice_type operator[](const std::size_t index)
		{
			// out-of-bounds ?
			assert(index < *_extents);
			assert(_offset + index * *_dimsMult < _data.size());
			return slice_type(_extents+1, _dimsMult+1, _data,
					  _offset + index * *_dimsMult);
		}

		const slice_type operator[](const std::size_t index) const
		{
			// out-of-bounds ?
			assert(index < *_extents);
			assert(_offset + index * *_dimsMult < _data.size());
			return slice_type(_extents+1, _dimsMult+1, _data,
					  _offset + index * *_dimsMult);
		}

		typename std::tr1::array<std::size_t, D>::const_iterator
		extents() const { return _extents; }

		iterator begin() const { return _data.begin() + _offset; }
		iterator end() const
		{ return _data.begin() + _offset + this->size(); }

		std::size_t size() const { return *_extents * *_dimsMult; }

	private:
		const typename
		std::tr1::array<std::size_t, D>::const_iterator _extents;
		const typename
		std::tr1::array<std::size_t, D>::const_iterator _dimsMult;
		DataHolder &_data;
		const std::size_t _offset;
};

template <typename T, class DataHolder>
class _RefMatrix<T, 1u, DataHolder>
{
	public:
		typedef T value_type;
		typedef typename DataHolder::iterator iterator;
		typedef typename DataHolder::const_iterator const_iterator;
		typedef typename DataHolder::pointer pointer;
		typedef typename DataHolder::reference reference;
		typedef typename DataHolder::const_reference const_reference;

		_RefMatrix(Matrix<T, 1u, DataHolder> &m) :
			_extents(m.extents()), _data(m._data), _offset(0) { }

		_RefMatrix(
		const std::tr1::array<std::size_t, 1u>::const_iterator &extents,
		const std::tr1::array<std::size_t, 1u>::const_iterator &dimsMult,
			DataHolder &data, const std::size_t offset) :
			_extents(extents), _data(data), _offset(offset)
		{
			assert(offset < _data.size());
			assert(offset + *_extents <= _data.size());
		}

		reference operator[] (const std::size_t index)
		{
			// out-of-bounds ?
			assert(index < *_extents);
			assert(_offset + index < _data.size());
// std::cout << "Index: " << _offset << " + " << index << " = " <<
// 	(_offset + index) << std::endl;

			return _data[_offset + index];
		}

		const reference operator[] (const std::size_t index) const
		{
			// out-of-bounds ?
			assert(index < *_extents);
			assert(_offset + index < _data.size());
// std::cout << "Index: " << _offset << " + " << index << " = " <<
// 	(_offset + index) << std::endl;

			return _data[_offset + index];
		}

		typename std::tr1::array<std::size_t, 1u>::const_iterator
		extents() const { return _extents; }

		iterator begin() const { return _data.begin() + _offset; }
		iterator end()   const { return _data.begin() + _offset + this->size(); }

		std::size_t size() const { return *_extents; }

	private:
		const typename
		std::tr1::array<std::size_t, 1u>::const_iterator _extents;
		DataHolder &_data;
		const std::size_t _offset;
};

template <typename T, std::size_t D,
	  class DataHolder = std::vector<T, AlignAllocator<T> > >
class MatrixImpl
{
	public:
		typedef T value_type;
		typedef typename DataHolder::iterator iterator;
		typedef typename DataHolder::const_iterator const_iterator;

//		typedef typename DataHolder::reference reference;
//		typedef typename DataHolder::const_reference const_reference;

		typedef _RefMatrix<T, D-1, DataHolder> slice_type;

		friend class _RefMatrix<T, D, DataHolder>;

		MatrixImpl(bool colMajor = false) : colMajor_(colMajor)
		{
			std::fill(_extents.begin(), _extents.end(), 0u);
			std::fill(_dimsMult.begin(), _dimsMult.end(), 1u);
		}

		template <typename Iterator>
		MatrixImpl(const Iterator extentsIt) : colMajor_(false),
			_data(std::accumulate(extentsIt, std::ptrdiff_t(D)+extentsIt, 1u,
					std::multiplies<std::size_t>()))
		{
			typedef typename
			std::iterator_traits<Iterator>::difference_type diff_t;
			std::copy(extentsIt, diff_t(D) + extentsIt, _extents.begin());

			this->computeDimsMult();
		}

		template <typename Iterator>
		MatrixImpl(const Iterator extentsIt, const T &value,
			const bool colMajor = false) : colMajor_(colMajor),
			_data(std::accumulate(extentsIt, extentsIt + D,
				1u, std::multiplies<std::size_t>()), value)
		{
			std::copy(extentsIt, extentsIt + D, _extents.begin());
			this->computeDimsMult();
		}

		template <typename Iterator1, typename Iterator2>
		MatrixImpl(const Iterator1 extentsIt,
			const Iterator2 beginData, const Iterator2 endData,
			bool colMajor = false) : colMajor_(colMajor),
				_data(beginData, endData)
		{
			std::copy(extentsIt, extentsIt + D, _extents.begin());
			this->computeDimsMult();

			// Did the user supply consistent information ?
			const std::size_t size = std::accumulate(extentsIt,
				extentsIt+D, 1u,std::multiplies<std::size_t>());
			if (_data.size() != size)
				_data.resize(size);
		}

		template <typename U, class DH>
		MatrixImpl &operator=(const MatrixImpl<U, D, DH> &m)
		{
			this->colMajor_ = m.colMajor();
			this->resize(m.extents());
			omptl::copy(m.begin(), m.end(), this->begin());
			return *this;
		}

		template <typename U>
		MatrixImpl &operator=(const U value)
		{
			omptl::fill(this->begin(), this->end(), value);
			return *this;
		}

		void changeMajor()
		{
			colMajor_ = !colMajor_;
			this->computeDimsMult();
		}

		std::size_t size() const
		{
			assert(std::accumulate(	_extents.begin(),
						_extents.begin() + D,
					1u, std::multiplies<std::size_t>())
				== _data.size());

			return _data.size();
		}

		typename std::tr1::array<std::size_t, D>::const_iterator
		extents() const { return _extents.begin(); }

		std::size_t extent(const std::size_t dimension) const
		{ assert(dimension < D); return _extents[dimension]; }

		template <typename Iterator>
		void resize(Iterator extentsIt, bool preserve = false)
		{
			assert(!preserve); // Not implemented
			std::copy(extentsIt, extentsIt + D, _extents.begin());
			const std::size_t newSize = std::accumulate(
					_extents.begin(), _extents.end(), 1u,
						std::multiplies<std::size_t>());
			if (newSize != _data.size())
				_data.resize(newSize);

			this->computeDimsMult();
		}

		slice_type operator[] (const std::size_t index)
		{
// std::cout << "Offset: " << index << " * " << _dimsMult[0u] << " = "
// 	<< (index * _dimsMult[0u]) << std::endl;

			// out-of-bounds ?
			assert(index < _extents[0]);
			return _RefMatrix<T, D-1u, DataHolder>
				(_extents.begin()+1, _dimsMult.begin()+1, _data,
				 index * _dimsMult[0u]);
		}

		const slice_type operator[] (const std::size_t index) const
		{
			// out-of-bounds ?
			assert(index < _extents[0]);
			return _RefMatrix<T, D-1u, DataHolder>
				(_extents.begin()+1, _dimsMult.begin()+1, _data,
				 index * _dimsMult[0u]);
		}

		typename DataHolder::iterator begin() { return _data.begin(); }
		typename DataHolder::iterator end() { return _data.end(); }

		typename DataHolder::const_iterator begin() const
		{ return _data.begin(); }
		typename DataHolder::const_iterator end() const
		{ return _data.end(); }

		std::size_t mem_size() const { return _data.size()*sizeof(T); }

		void swap(MatrixImpl &that)
		{
			std::swap(this->colMajor_, that.colMajor_);
			this->_extents.swap(that._extents);
			this->_dimsMult.swap(that._dimsMult);
			this->_data.swap(that._data);
		}

		void clear()
		{
			_data.clear();
			std::fill(_extents.begin(), _extents.end(), 0);
		}

		friend bool operator==(const MatrixImpl &lhs,
					const MatrixImpl &rhs)
		{
			if (lhs.size() != rhs.size())
				return false;

			if (!omptl::equal(lhs.extents(), lhs.extents()+D,
					  rhs.extents()))
				return false;

			return omptl::equal(lhs.begin(), lhs.end(),rhs.begin());
		}

		bool colMajor() const { return colMajor_; }

	private:
		bool colMajor_;
		std::tr1::array<std::size_t, D> _extents;
		std::tr1::array<std::size_t, D> _dimsMult;
		mutable DataHolder _data;

		void computeDimsMult()
		{
			if (colMajor_)
			{
				_dimsMult[0] = _extents[D-1];
				for (std::size_t i = 1; i < D; ++i)
					_dimsMult[i] = _extents[D-1-i] *
							_dimsMult[i-1u];
			}
			else
			{
				_dimsMult[D-1] = 1u;
				for (int i = D-2; i >= 0; --i)
					_dimsMult[i] = _extents[i+1u] *
							_dimsMult[i+1u];
			}
		}
};

template <typename T, class DataHolder>
class MatrixImpl<T, 1u, DataHolder>
{
	public:
		typedef T value_type;
		typedef typename DataHolder::iterator iterator;
		typedef typename DataHolder::const_iterator const_iterator;
		typedef typename DataHolder::reference reference;
		typedef typename DataHolder::const_reference const_reference;

		friend class _RefMatrix<T, 1u, DataHolder>;

		MatrixImpl() { _extents[0] = 0u; }

		template <typename Iterator>
		MatrixImpl(Iterator extentsIt) : _data(*extentsIt)
		{ _extents[0] = *extentsIt; }

		template <typename Iterator>
		MatrixImpl(Iterator extentsIt, const T &value) :
			_data(*extentsIt, value)
		{ _extents[0] = *extentsIt; }

		std::size_t size() const
		{ return _extents[0]; }

		std::tr1::array<std::size_t, 1u>::const_iterator extents() const
		{ return _extents.begin(); }

		std::size_t extent(const std::size_t dimension) const
		{ assert(dimension < 1); return _extents[0]; }

		template <typename Iterator>
		void resize(Iterator extentsIt, bool preserve = false)
		{
			const std::size_t newSize = *extentsIt;
			if (newSize != _data.size())
			{
				_extents[0] = newSize;
				_data.resize(newSize);
			}
		}

		reference operator[] (const std::size_t index)
		{
			// out-of-bounds ?
			assert(index < _extents[0]);
			return _data[index];
		}

		const_reference operator[](const std::size_t index) const
		{
			// out-of-bounds ?
			assert(index < _extents[0]);
			return _data[index];
		}

		typename DataHolder::iterator begin() { return _data.begin(); }
		typename DataHolder::iterator end() { return _data.end(); }

		typename DataHolder::const_iterator begin() const
		{ return _data.begin(); }
		typename DataHolder::const_iterator end() const
		{ return _data.end(); }

		std::size_t mem_size() const
		{ return _data.capacity() * sizeof(T); }

		void swap(MatrixImpl &m)
		{
			this->_extents.swap(m._extents);
			this->_data.swap(m._data);
		}

		void clear() { _data.clear(); _extents[0] = 0; }

// 		template <U, DHL, DHR>
		friend bool operator==(const MatrixImpl &lhs,
					const MatrixImpl &rhs)
		{
			if (lhs.size() != rhs.size())
				return false;

			return omptl::equal(lhs.begin(), lhs.end(),rhs.begin());
		}

	private:
		std::tr1::array<std::size_t, 1u> _extents;
		DataHolder _data;
};

template <typename T, std::size_t D,
	  class DataHolder = std::vector<T, AlignAllocator<T> > >
class Matrix
{
	public:
		typedef T value_type;
		typedef typename MatrixImpl<T,D,DataHolder>::iterator iterator;
		typedef typename MatrixImpl<T,D,DataHolder>::const_iterator const_iterator;
		typedef typename MatrixImpl<T,D,DataHolder>::slice_type	slice_type;
		typedef Matrix* pointer;

		typedef Matrix& reference;
		typedef const Matrix& const_reference;

		explicit Matrix(bool colMajor = false) :
			matrix_(new MatrixImpl<T, D, DataHolder>(colMajor))
		{ }

		template <typename Iterator>
		explicit Matrix(const Iterator extentsIt) :
			matrix_(new MatrixImpl<T, D, DataHolder>(extentsIt))
		{ }


		template <typename Iterator>
		explicit Matrix(const Iterator extentsIt, const T &value,
			const bool colMajor = false) :
			matrix_(new MatrixImpl<T,D,DataHolder>(extentsIt,
							value, colMajor))
		{ }

		template <typename Iterator1, typename Iterator2>
		explicit Matrix(const Iterator1 extentsIt,
			const Iterator2 beginData, const Iterator2 endData,
			bool colMajor = false) :
			matrix_(new MatrixImpl<T,D,DataHolder>(extentsIt,
					beginData, endData, colMajor))
		{ }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
		template <typename U>
		explicit Matrix(const std::initializer_list<U> init_list) :
			matrix_(new MatrixImpl<T,D,DataHolder>(init_list.begin()))
		{ }
		template <typename U>
		explicit Matrix(const std::initializer_list<U> init_list, const T &value,
			const bool colMajor = false) :
			matrix_(new MatrixImpl<T,D,DataHolder>(init_list.begin(),
							value, colMajor))
		{ }

		template <typename U, typename Iterator2>
		explicit Matrix(const std::initializer_list<U> init_list,
			const Iterator2 beginData, const Iterator2 endData,
			bool colMajor = false) :
			matrix_(new MatrixImpl<T,D,DataHolder>(init_list.begin(),
					beginData, endData, colMajor))
		{ }
#endif

		Matrix(const Matrix &m) :
			matrix_(m.matrix_)
		{ /*assert(this->matrix_ == m.matrix_);*/ }

		Matrix &operator=(const Matrix &m)
		{
			this->matrix_ = m.matrix_;
			//assert(this->matrix_ == m.matrix_);
			return *this;
		}

		template <typename U, class DH>
		Matrix &operator=(const Matrix<U, D, DH> &m)
		{
			*this->matrix_ = *m.matrix_;
			return *this;
		}

		template <typename U>
		Matrix &operator=(const U value)
		{
			*matrix_ = value;
			return *this;
		}

		Matrix clone() const
		{
			Matrix newMatrix;
			newMatrix.matrix_ =
			std::tr1::shared_ptr<MatrixImpl<T,D,DataHolder> >
				(new MatrixImpl<T,D,DataHolder>(*matrix_));
			return newMatrix;
		}

		void changeMajor() { matrix_->changeMajor(); }

		std::size_t size() const { return matrix_->size(); }

		typename std::tr1::array<std::size_t, D>::const_iterator
		extents() const { return matrix_->extents(); }

		std::size_t extent(const std::size_t dimension) const
		{ return matrix_->extent(dimension); }

		template <typename Iterator>
		void resize(Iterator extentsIt, bool preserve = false)
		{ matrix_->resize(extentsIt, preserve); }

		slice_type operator[] (const std::size_t index)
		{ return (*matrix_)[index]; }

		const slice_type operator[] (const std::size_t index) const
		{
			return static_cast<const MatrixImpl<T, D, DataHolder>& >
				(*matrix_)[index];
		}

		typename DataHolder::iterator begin() { return matrix_->begin(); }
		typename DataHolder::iterator end()   { return matrix_->end();   }

		typename DataHolder::const_iterator begin() const
		{ return matrix_->begin(); }
		typename DataHolder::const_iterator end() const
		{ return matrix_->end(); }

		std::size_t mem_size() const { return matrix_->mem_size(); }

		void swap(Matrix &m) { std::swap(this->matrix_, m.matrix_); }

		void clear() { matrix_->clear(); }

		friend bool operator==(const Matrix &lhs, const Matrix &rhs)
		{
			if (lhs.matrix_ == rhs.matrix_)
				return true;

			return *lhs.matrix_ == *rhs.matrix_;
		}

		friend bool operator!=(const Matrix &lhs,
					const Matrix &rhs)
		{ return !(lhs == rhs); }

		bool colMajor() const { return matrix_->colMajor(); }

	private:
		mutable std::tr1::shared_ptr<MatrixImpl<T,D,DataHolder> > matrix_;
};


template <typename T, class DataHolder>
class Matrix<T, 1u, DataHolder>
{
	public:
		typedef T value_type;
		typedef typename MatrixImpl<T,1u,DataHolder>::iterator iterator;
		typedef typename MatrixImpl<T, 1u, DataHolder>::const_iterator
								const_iterator;
		typedef typename MatrixImpl<T, 1u, DataHolder>::reference
								reference;
		typedef typename MatrixImpl<T, 1u, DataHolder>::const_reference
								const_reference;

		Matrix() : matrix_(new MatrixImpl<T , 1u, DataHolder>()) { }

		template <typename Iterator>
		Matrix(const Iterator extentsIt) :
			matrix_(new MatrixImpl<T, 1u,DataHolder>(extentsIt))
		{ }

		template <typename Iterator>
		Matrix(const Iterator extentsIt, const T &value,
			const bool colMajor = false) :
			matrix_(new MatrixImpl<T, 1u,DataHolder>(extentsIt, value))
		{ }

		template <typename Iterator1, typename Iterator2>
		Matrix(const Iterator1 extentsIt,
			const Iterator2 beginData, const Iterator2 endData,
			bool colMajor = false) :
			matrix_(new MatrixImpl<T, 1u, DataHolder>(extentsIt,
					beginData, endData))
		{ }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
		template <typename U>
		explicit Matrix(const std::initializer_list<U> init_list,
				const bool colMajor = false) :
			matrix_(new MatrixImpl<T,1u,DataHolder>(init_list.begin()))
		{ }
		template <typename U>
		explicit Matrix(const std::initializer_list<U> init_list,
				const T &value, const bool colMajor = false) :
			matrix_(new MatrixImpl<T,1u,DataHolder>(init_list.begin(), value))
		{ }

		template <typename U, typename Iterator>
		explicit Matrix(const std::initializer_list<U> init_list,
			const Iterator beginData, const Iterator endData,
			bool colMajor = false) :
			matrix_(new MatrixImpl<T,1u,DataHolder>(init_list.begin(),
					beginData, endData))
		{ }
#endif

		Matrix(const Matrix &m) :
			matrix_(m.matrix_)
		{ assert(this->matrix_ == m.matrix_); }

		Matrix &operator=(const Matrix &m)
		{
			this->matrix_ = m.matrix_;
			assert(this->matrix_ == m.matrix_);
			return *this;
		}

		template <typename U, class DH>
		Matrix &operator=(const Matrix<U, 1u, DH> &m)
		{
			*this->matrix_ = *m.matrix_;
			return *this;
		}

		template <typename U>
		Matrix &operator=(const U value)
		{
			*matrix_ = value;
			return *this;
		}

		Matrix clone() const
		{
			Matrix newMatrix;
			newMatrix.matrix_ =
			std::tr1::shared_ptr<MatrixImpl<T, 1u, DataHolder> >
				(new MatrixImpl<T, 1u, DataHolder>(*matrix_));
			return newMatrix;
		}

// 		void changeMajor() { matrix_->changeMajor(); }

		std::size_t size() const { return matrix_->size(); }

		typename std::tr1::array<std::size_t, 1u>::const_iterator
		extents() const { return matrix_->extents(); }

		std::size_t extent(const std::size_t dimension) const
		{ return matrix_->extent(dimension); }

		template <typename Iterator>
		void resize(Iterator extentsIt, bool preserve = false)
		{ matrix_->resize(extentsIt, preserve); }

		reference operator[] (const std::size_t index)
		{ return (*matrix_)[index]; }

		const_reference operator[] (const std::size_t index) const
		{
			return static_cast<const MatrixImpl<T, 1u, DataHolder>&>
				(*matrix_)[index];
		}

		typename DataHolder::iterator begin() {return matrix_->begin();}
		typename DataHolder::iterator end() { return matrix_->end(); }

		typename DataHolder::const_iterator begin() const
		{ return matrix_->begin(); }
		typename DataHolder::const_iterator end() const
		{ return matrix_->end(); }

		std::size_t mem_size() const { return matrix_->mem_size(); }

		void swap(Matrix &m) { std::swap(this->matrix_, m.matrix_); }

		void clear() { matrix_->clear(); }

		friend bool operator==(const Matrix &lhs, const Matrix &rhs)
		{
			if (lhs.matrix_ == rhs.matrix_)
				return true;

			return *lhs.matrix_ == *rhs.matrix_;
		}

		friend bool operator!=(const Matrix &lhs, const Matrix &rhs)
		{ return !(lhs == rhs); }

	private:
		std::tr1::shared_ptr<MatrixImpl<T, 1, DataHolder> > matrix_;
};

template <typename T, std::size_t D>
class CompressedMatrix : public Matrix<T, D, CompressDataHolder<T> >
{
	private:
		typedef Matrix<T, D, CompressDataHolder<T> > Super;
	public:
		typedef typename Super::iterator iterator;
		typedef typename Super::const_iterator const_iterator;
		typedef T value_type;

		CompressedMatrix()
		{ }

		template <typename Iterator>
		CompressedMatrix(Iterator extentsIt) : Super(extentsIt)
		{ }

		template <typename Iterator>
		CompressedMatrix(Iterator extentsIt, const value_type &value) :
			 Super(extentsIt, value)
		{ }
};

template <typename T>
class _DynamicMatrixRef
{
	public:
		typedef T value_type;
		typedef _DynamicMatrixRef<T> slice_type;

		_DynamicMatrixRef(
			std::vector<std::size_t>::const_iterator extents,
			std::vector<std::size_t>::const_iterator size,
			std::vector<T> &data,
			 std::size_t dimensions, std::size_t offset) :
				_extents(extents), _size(size), _data(data),
				_dimensions(dimensions), _offset(offset) { }

		std::size_t size() const { return *_size; }

		std::size_t dimensions() const { return _dimensions; }

		operator T&()
		{
			assert(_dimensions == 0u);
			assert(_offset < _data.size());
			return _data[_offset];
		}

		operator const T&() const
		{
			assert(_dimensions == 0u);
			assert(_offset < _data.size());
			return _data[_offset];
		}

		T operator=(const T &d)
		{
			assert(_dimensions == 0u);
			_data[_offset] = d;
			return d;
		}

		_DynamicMatrixRef<T> operator[](std::size_t index)
		{
			assert(_dimensions > 0u);
			// out-of-bounds ?
			assert(index < *_extents);
			return _DynamicMatrixRef(_extents + 1, _size + 1,
				_data, _dimensions - 1u,
				_offset + index * _size[1]);
		}

		const _DynamicMatrixRef<T> operator[](std::size_t index) const
		{
			assert(_dimensions > 0u);
			// out-of-bounds ?
			assert(index < *_extents);
			return _DynamicMatrixRef(_extents + 1, _size + 1,
				_data, _dimensions - 1u,
				_offset + index * _size[1]);
		}

		typename std::vector<T>::iterator begin()
		{ return _data.begin() + _offset; }
		typename std::vector<T>::iterator end()
		{ return _data.begin() + _offset + this->size(); }

		typename std::vector<T>::const_iterator begin() const
		{ return _data.begin() + _offset; }
		typename std::vector<T>::const_iterator end() const
		{ return _data.begin() + _offset + this->size(); }

	private:
		std::vector<std::size_t>::const_iterator _extents;
		std::vector<std::size_t>::const_iterator _size;
		std::vector<T> &_data;
		const std::size_t _dimensions;
		const std::size_t _offset;
};

template <typename T>
class DynamicMatrixImpl
{
	public:
		typedef T value_type;
		typedef _DynamicMatrixRef<T> slice_type;

		template <typename It>
		DynamicMatrixImpl(It extentsBegin, It extentsEnd) :
			_extents(extentsBegin, extentsEnd),
			_data(std::accumulate(extentsBegin, extentsEnd,
				1u, std::multiplies<std::size_t>())),
			_size(_extents.size()+1)
		{
			assert(this->dimensions() > 0u);
			_size[this->dimensions()] = 1;
			for (std::size_t i = this->dimensions(); i > 0; --i)
				_size[i-1] =_extents[i-1] * _size[i];
		}

		template <typename It>
		DynamicMatrixImpl(It extentsBegin, It extentsEnd, T value) :
			_extents(extentsBegin, extentsEnd),
			_data(std::accumulate(extentsBegin, extentsEnd,
				1u, std::multiplies<std::size_t>()), value),
			_size(_extents.size()+1)
		{
			assert(this->dimensions() > 0u);
			_size[this->dimensions()] = 1;
			for (std::size_t i = this->dimensions(); i > 0; --i)
				_size[i-1] =_extents[i-1] * _size[i];
		}

		std::size_t size() const { return _size[0]; }

		std::size_t dimensions() const { return _extents.size(); }

		std::vector<std::size_t>::const_iterator extents() const
		{ return _extents.begin(); }

		_DynamicMatrixRef<T> operator[](std::size_t index)
		{
			assert(this->dimensions() > 1u);
			// out-of-bounds ?
			assert(index < _extents[this->dimensions() - 1u]);
			return _DynamicMatrixRef<T>(_extents.begin() + 1,
					_size.begin() + 1, _data,
					this->dimensions() - 1u,
					index * _size[1]);
		}

		const _DynamicMatrixRef<T> operator[](std::size_t index) const
		{
			assert(this->dimensions() > 1u);
			return _DynamicMatrixRef<T>(_extents.begin(),
					_size.begin(), _data,
					this->dimensions() - 1u,
					index * _size[1]);
		}

		template <typename It>
		void resize(It extentsBegin, It extentsEnd,
			    bool preserve = false)
		{
			assert(!preserve); // Not implemented

			_extents.assign(extentsBegin, extentsEnd);
			assert(this->dimensions() > 0u);

			_size.resize(this->dimensions() + 1u);
			_size[this->dimensions()] = 1u;
			for (std::size_t i = this->dimensions(); i > 0; --i)
				_size[i-1u] =_extents[i-1u] * _size[i];

			const std::size_t newSize = _size[this->dimensions()];
			if (newSize != _data.size())
				_data.resize(newSize);
		}

		typename std::vector<T>::iterator begin()
		{ return _data.begin(); }
		typename std::vector<T>::iterator end() { return _data.end(); }

		typename std::vector<T>::const_iterator begin() const
		{ return _data.begin(); }
		typename std::vector<T>::const_iterator end() const
		{ return _data.end(); }

		void swap(DynamicMatrixImpl &that)
		{
			this->_extents.swap(that._extents);
			this->_data.swap(that._data);
			this->_size.swap(that._size);
		}

		void clear()
		{
			this->_data.clear();
			std::fill(_extents.begin(), _extents.end(), 0);
			std::fill(_size.begin(), _size.end(), 0);
		}

	private:
		std::vector<std::size_t> _extents;
		std::vector<T> _data;
		std::vector<std::size_t> _size;
};

template <typename T>
class DynamicMatrix
{
	public:
		typedef typename _DynamicMatrixRef<T>::value_type value_type;
		typedef typename _DynamicMatrixRef<T>::slice_type slice_type;

		template <typename It>
		DynamicMatrix(It extentsBegin, It extentsEnd) :
			matrix_(new DynamicMatrixImpl<T>(extentsBegin,
							 extentsEnd))
		{ }

		template <typename It>
		DynamicMatrix(It extentsBegin, It extentsEnd, T value) :
			matrix_(new DynamicMatrixImpl<T>(extentsBegin,
							 extentsEnd, value))
		{ }

		std::size_t size() const { return matrix_->size(); }

		std::size_t dimensions() const { return matrix_->dimensions(); }

		std::vector<std::size_t>::const_iterator extents() const
		{ return matrix_->extents(); }

		_DynamicMatrixRef<T> operator[](std::size_t index)
		{ return (*matrix_)[index]; }

		const _DynamicMatrixRef<T> operator[](std::size_t index) const
		{
			return static_cast<const DynamicMatrixImpl<T> & >
					(*matrix_)[index];
		}

		template <typename It>
		void resize(It extentsBegin, It extentsEnd,
			    bool preserve = false)
		{ matrix_->resize(extentsBegin, extentsEnd, preserve); }

		typename std::vector<T>::iterator begin()
		{ return matrix_->begin(); }
		typename std::vector<T>::iterator end() {return matrix_->end();}

		typename std::vector<T>::const_iterator begin() const
		{ return matrix_->begin(); }
		typename std::vector<T>::const_iterator end() const
		{ return matrix_->end(); }

		void swap(DynamicMatrix &that)
		{ std::swap(this->matrix_, that.matrix_); }

		void clear() { matrix_->clear(); }

	private:
		std::tr1::shared_ptr<DynamicMatrixImpl<T> > matrix_;
};

} //  end namespace cvmlcpp

/********************
 * SYMMETRIC MATRIX *
 ********************/

// Forward declaration
namespace cvmlcpp{ namespace detail { template <typename T> class SymmetricMatrixConstIterator; } }

template <typename T>
bool operator!=( const cvmlcpp::detail::SymmetricMatrixConstIterator<T> &lhs,
		 const cvmlcpp::detail::SymmetricMatrixConstIterator<T> &rhs)
{
	return lhs.index_ != rhs.index_ || lhs.m_ != rhs.m_;
}

template <typename T>
bool operator==( const cvmlcpp::detail::SymmetricMatrixConstIterator<T> &lhs,
		 const cvmlcpp::detail::SymmetricMatrixConstIterator<T> &rhs)
{ return !(lhs != rhs); }

namespace cvmlcpp
{

/*
 * Iterators
 */
namespace detail
{

template <typename T>
class SymmetricMatrixImpl;

template <typename T>
class SymmetricMatrixIterator
{
	public:
		typedef T value_type;
		typedef std::ptrdiff_t difference_type;
		typedef std::random_access_iterator_tag iterator_category;
		typedef typename std::vector<T>::pointer pointer;
		typedef typename std::vector<T>::reference reference;

		reference operator*() const
		{
			assert(index_ >= 0);
			assert(std::size_t(index_) < m_.rank()*m_.rank());
			const std::ptrdiff_t r = index_ / m_.rank();
			const std::ptrdiff_t c = index_ % m_.rank();

			return m_(r, c);
		}

		SymmetricMatrixIterator operator++()
		{ ++this->index_; return *this; }

		SymmetricMatrixIterator operator++(int)
		{
			SymmetricMatrixIterator it(*this);
			++(*this);
			return it;
		}

		template <typename U>
		SymmetricMatrixIterator operator+=(const U v)
		{ this->index_ += v; return *this; }

		template <typename U>
		SymmetricMatrixIterator operator-=(const U v)
		{ this->index_ -= v; return *this; }

		template <typename U>
		SymmetricMatrixIterator operator+(const U v) const
		{ return SymmetricMatrixIterator(const_cast<SymmetricMatrixIterator<T> *>(this)->m_, this->index_+v); }

		template <typename U>
		SymmetricMatrixIterator operator-(const U v) const
		{ return SymmetricMatrixIterator(const_cast<SymmetricMatrixIterator<T> *>(this)->m_, this->index_-v); }

		operator std::ptrdiff_t() const { return index_; }

	private:
		SymmetricMatrix<T> m_;
		std::ptrdiff_t index_;
		friend class SymmetricMatrixConstIterator<T>;
		friend class SymmetricMatrix<T>;
		SymmetricMatrixIterator(SymmetricMatrix<T> &m, const std::ptrdiff_t index = 0) :
			m_(m), index_(index) { }
};

template <typename T>
class SymmetricMatrixConstIterator
{
	public:
		typedef T value_type;
		typedef std::ptrdiff_t			difference_type;
		typedef std::random_access_iterator_tag iterator_category;
		typedef const T* pointer;
		typedef const T& reference;

		SymmetricMatrixConstIterator(const SymmetricMatrixIterator<T> &it) :
			m_(it.m_), index_(it.index_) { }

		reference operator*() const
		{
			assert(index_ >= 0);
			assert(std::size_t(index_) < m_.rank()*m_.rank());
			const std::ptrdiff_t r = index_ / m_.rank();
			const std::ptrdiff_t c = index_ % m_.rank();

			return m_(r, c);
		}

		SymmetricMatrixConstIterator operator++()
		{ ++this->index_; return *this; }

		SymmetricMatrixConstIterator operator++(int)
		{
			SymmetricMatrixConstIterator it(*this);
			++(*this);
			return it;
		}

		template <typename U>
		SymmetricMatrixConstIterator operator+=(const U v)
		{ this->index_ += v; return *this; }

		template <typename U>
		SymmetricMatrixConstIterator operator+(const U v) const
		{ return SymmetricMatrixConstIterator(this->m_, this->index_+v); }

		template <typename U>
		SymmetricMatrixConstIterator operator-(const U v) const
		{ return SymmetricMatrixConstIterator(this->m_, this->index_-v); }

		operator std::ptrdiff_t() const { return index_; }

	private:
		const SymmetricMatrix<T> m_;
		std::ptrdiff_t index_;
//		friend bool operator!=<>(const SymmetricMatrixConstIterator<T> &lhs,
//					 const SymmetricMatrixConstIterator<T> &rhs);
		template <class U> friend bool operator!=(const SymmetricMatrixConstIterator<U> &lhs,
					 const SymmetricMatrixConstIterator<U> &rhs);
		friend class SymmetricMatrix<T>;
		SymmetricMatrixConstIterator(const SymmetricMatrix<T> &m, const std::ptrdiff_t index = 0) :
			m_(m), index_(index) { }
};

template <typename T>
class SymmetricMatrixImpl
{
	public:
		typedef T value_type;
		typedef typename detail::SymmetricMatrixIterator<T> iterator;
		typedef typename detail::SymmetricMatrixConstIterator<T> const_iterator;
		typedef typename std::vector<T>::reference reference;
		typedef typename std::vector<T>::iterator row_iterator;
		typedef typename std::vector<T>::const_iterator const_row_iterator;

		SymmetricMatrixImpl(const std::size_t rank = 0) :
		n_(rank), data_(rank*(rank+1u)/2u) { }

		SymmetricMatrixImpl(const std::size_t rank, const T value) :
		n_(rank), data_(rank*(rank+1u)/2u, value) { }

		void resize(const std::size_t rank)
		{ n_ = rank; data_.resize( rank*(rank+1u)/2u ); }

		std::size_t size() const { return data_.size(); }
		std::size_t rank() const
		{
			// verify with ABC-formula
			assert(std::sqrt(1.f+8.f*data_.size()) == n_*2.f + 1.f);
			return n_;
		}

		const SymmetricMatrixImpl operator=(const T value)
		{
			std::fill(data_.begin(), data_.end(), value);
			return *this;
		}
		/*
		 * Iterators per row
		 */
		row_iterator begin(const std::size_t row)
		{ return data_.begin() + this->index(row, row); }

		row_iterator end(const std::size_t row)
		{ return this->begin(row) + n_-row; }

		const_row_iterator begin(const std::size_t row) const
		{ return data_.begin() + this->index(row, row); }

		const_row_iterator end(const std::size_t row) const
		{ return this->begin(row) + n_-row; }

		// Individual data access
		reference operator()(const std::size_t row, const std::size_t column)
		{ return data_[this->index(row, column)]; }

		const reference operator()(const std::size_t row,
					   const std::size_t column) const
		{ return data_[this->index(row, column)]; }
/*
		void swap(SymmetricMatrixImpl &that)
		{
			std::swap(this->n_, that.n_);
			this->data_.swap(that.data_);
		}
*/
	private:
		friend class detail::SymmetricMatrixIterator<T>;
		friend class detail::SymmetricMatrixConstIterator<T>;
		std::size_t n_;
		std::vector<T> data_;

		std::size_t index(const std::size_t row,
				  const std::size_t column) const
		{
			assert( (n_+1)*n_/2 == data_.size() );

			const std::size_t r = std::min(row, column);
			const std::size_t c = std::max(row, column);
			assert(r <= c);
			assert(r < n_);
			assert(c <= n_); // inclusive to find index of end() element

			// Offset: index of the diagonal element of row r
			// in data_
			const std::size_t offset = (2u*n_-r+1)*(r/2u) +
					( (r&1) ? (n_-(r-1u)/2u) : 0 );
			assert(offset + c-r < data_.size());

			return offset + c-r;
		}
};

} // end namespace detail

template <typename T>
class SymmetricMatrix
{
	public:
		typedef T value_type;
		typedef typename detail::SymmetricMatrixImpl<T>::iterator iterator;
		typedef typename detail::SymmetricMatrixImpl<T>::const_iterator const_iterator;
		typedef typename detail::SymmetricMatrixImpl<T>::reference reference;
		typedef typename detail::SymmetricMatrixImpl<T>::iterator row_iterator;
		typedef typename detail::SymmetricMatrixImpl<T>::const_iterator const_row_iterator;

		SymmetricMatrix(const std::size_t rank = 0) :
			p(new detail::SymmetricMatrixImpl<T>(rank)) { assert(p); }

		SymmetricMatrix(const std::size_t rank, const T value) :
			p(new detail::SymmetricMatrixImpl<T>(rank, value)) { assert(p);}

		void resize(const std::size_t rank) { assert(p); p->resize(rank); }

		std::size_t size() const { return p->size(); }
		std::size_t rank() const { return p->rank(); }

		// Iterators over all data
		iterator begin() { return iterator(*this); }
		iterator end()   { return iterator(*this, this->rank()*this->rank()); }
		const_iterator begin() const { return const_iterator(*this); }
		const_iterator end()   const { return const_iterator(*this,  this->rank()*this->rank()); }

		const SymmetricMatrix operator=(const T value)
		{
			*p = value;
			return *this;
		}

		/*
		 * Iterators per row
		 */
		row_iterator begin(const std::size_t row)
		{ return p->begin(row); }

		row_iterator end(const std::size_t row)
		{ return p->end(row); }

		const_row_iterator begin(const std::size_t row) const
		{ return p->begin(row); }

		const_row_iterator end(const std::size_t row) const
		{ return p->end(row); }

		// Individual data access
		reference operator()(const std::size_t row, const std::size_t column)
		{ return (*p)(row, column); }

		const reference operator()(const std::size_t row,
					   const std::size_t column) const
		{ return (*p)(row, column); }

		void swap(SymmetricMatrix &that) { std::swap(this->p, that.p); }

	private:
		std::tr1::shared_ptr<detail::SymmetricMatrixImpl<T> > p;
};


} // namespace cvmlcpp

/*
 * Swap functions as recommended by Effective C++
 */

namespace std
{

template <typename T, std::size_t D, class DataHolder>
void swap(cvmlcpp::Matrix<T, D, DataHolder> &lhs,
	  cvmlcpp::Matrix<T, D, DataHolder> &rhs)
{
	lhs.swap(rhs);
}

template <typename T>
void swap(cvmlcpp::DynamicMatrix<T> &lhs, cvmlcpp::DynamicMatrix<T> &rhs)
{
	lhs.swap(rhs);
}

template <typename T, const std::size_t D>
void swap(cvmlcpp::CompressedMatrix<T, D> &lhs,
	  cvmlcpp::CompressedMatrix<T, D> &rhs)
{
	lhs.swap(rhs);
}

template <typename T>
void swap(cvmlcpp::SymmetricMatrix<T> &lhs, cvmlcpp::SymmetricMatrix<T> &rhs)
{
	lhs.swap(rhs);
}

} // namespace std
/*
template <typename T>
bool operator!=( const cvmlcpp::detail::SymmetricMatrixConstIterator<T> &lhs,
		 const cvmlcpp::detail::SymmetricMatrixConstIterator<T> &rhs)
{
	return lhs.index_ != rhs.index_ || lhs.m_ != rhs.m_;
}
*/
