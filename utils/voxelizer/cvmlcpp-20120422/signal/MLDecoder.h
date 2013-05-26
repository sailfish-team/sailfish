/***************************************************************************
 *   Copyright (C) 2008 by BEEKHOF, Fokko                                  *
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

#ifndef CVMLCPP_MLDECODER_H
#define CVMLCPP_MLDECODER_H 1

#include <functional>

#include <omptl/omptl_algorithm>

#include <cvmlcpp/base/StringTools>

namespace cvmlcpp
{

class _VarConfidence
{
	public:
		void init(const std::size_t id, const double p)
		{
			_id = id;
			_p  = p;
		}

		bool operator<(const _VarConfidence &that) const
		{ return std::abs(this->p() - 0.5) < std::abs(that.p() - 0.5); }

		std::size_t id() const { return _id; }

		const double p() const { return _p; }

	private:
		std::size_t _id;
		double _p;
};

class MLVariable_
{
	public:
		typedef std::vector<std::size_t>::const_iterator const_iterator;

		void init(const double p, const double soft = 0.0)
		{
			_p = std::max(soft, std::min(1.0-soft, p));
			_isSet = false;
			_value = mlValue();
		}

		bool isSet() const { return _isSet; }

		bool value() const
		{
			return this->_value;
		}

		bool value(bool mostLikely) const
		{
			return mostLikely ? mlValue() : !mlValue();
		}

		double likelihood(const bool val) const
		{
			return val ? _p : (double(1)-_p);
		}

		bool mlValue() const { return _p > 0.5; }

		double score(bool mostLikely) const
		{
			if (mostLikely)
				return std::log(likelihood(mlValue()));
			return std::log(likelihood(!mlValue()));
		}

		double score() const
		{ return std::log(likelihood(this->value())); }

		void set(const bool mostLikely)
		{
			assert(!_isSet);
			_isSet = true;
			_value = mostLikely?(_p>0.5):(_p<=0.5);
		}

		void unset()
		{
			assert(_isSet);
			_isSet = false;
			_value = _p > 0.5;
		}

		void addLink(const std::size_t chkID)
		{
// 			assert(!_lock);
			assert(std::find(_chkIDs.begin(), _chkIDs.end(), chkID)
				== _chkIDs.end());
			_chkIDs.push_back(chkID);
		}

		const_iterator begin() const { return _chkIDs.begin(); }
		const_iterator end()   const { return _chkIDs.end(); }

		const double p() const { return _p; }

	private:
		double _p;
		bool _isSet;
		bool _value;
		std::vector<std::size_t> _chkIDs;
};

class MLCheck_
{
	public:
		typedef std::vector<std::size_t>::const_iterator const_iterator;

		 // FIXME more elegant solution needed
		MLCheck_() : _nrFreeVars(-1) { assert(_varIDs.size() == 0u); }

		void addLink(const std::size_t varID)
		{
// 			assert(!_lock);
			assert(std::find(_varIDs.begin(), _varIDs.end(), varID)
				== _varIDs.end());
			_varIDs.push_back(varID);
			_nrFreeVars = _varIDs.size();
		}

		void reset()
		{
// std::cout << _varIDs.size() << " == " << _nrFreeVars << std::endl;
			assert(_nrFreeVars == _varIDs.size());
			_nrFreeVars = _varIDs.size();
		}

		std::size_t size() const { return _varIDs.size(); }

		template <std::size_t N>
		bool satisfied(std::tr1::array<MLVariable_, N> &vars) const
		{
// 			assert(_lock);

			bool ok = true;
// 			#pragma omp parallel for
			for (int i = 0; i < int(_varIDs.size()); ++i)
				ok ^= vars[_varIDs[i]].value();

			return ok;
		}

		std::size_t nrFreeVars() const
		{ /* assert(_lock); */ return _nrFreeVars;}

// 		void lock() { assert(!_lock);  _lock = true; }

		// Returns remaining number of free vars
		std::size_t setVar(const std::size_t varID)
		{
// 			assert(_lock);
			assert(std::find(_varIDs.begin(), _varIDs.end(), varID)
				!= _varIDs.end());
			assert(nrFreeVars() > 0u);
			return --_nrFreeVars;
		}

		std::size_t unsetVar(const std::size_t varID)
		{
			assert(nrFreeVars() < _varIDs.size());
// 			assert(_lock);
			assert(std::find(_varIDs.begin(), _varIDs.end(), varID)
				!= _varIDs.end());
			return ++_nrFreeVars;
		}

		const_iterator begin() const { return _varIDs.begin(); }
		const_iterator end()   const { return _varIDs.end(); }

	private:
// 		bool _lock;
		std::size_t _nrFreeVars;
		std::vector<std::size_t> _varIDs;
};

template <std::size_t NVariables, std::size_t NChecks>
class MLDecoder
{
	public:
		template <typename Iterator>
		bool correct(Iterator first, const Iterator last,
				std::bitset<NVariables> &codeword)
		{
// std::cout << "MLDecoder::correct()" << std::endl;
// this->print();
#ifndef NDEBUG
			nodes = 0;
#endif
			depth.clear();
			depth.push_back(0);
			freeVars.set();
			assert(freeVars.size()  == NVariables);
			assert(freeVars.count() == NVariables);
			vids.clear();

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (int i = 0; i < int(chks.size()); ++i)
			{
				chks[i].reset();
				assert(chks[i].nrFreeVars() > 0u);
				assert(chks[i].nrFreeVars() < NVariables);
			}

// 			omptl::for_each(chks.begin(), chks.end(),
// 				std::mem_fun(&cvmlcpp::MLCheck_::reset));

			assert(std::distance(first, last) == NVariables);
			for (std::size_t i = 0; first != last; ++i, ++first)
			{
				assert(i < vars.size());
				vars[i].init(*first);
				varConfidences[i].init(i, *first);
			}

			// Calculate score of current assignment. The current
			// assignment does not correspond to a valid codeword
			// (otherwise we wouldn't be decoding). As the
			// assignment is changed, the score is adjusted.
			double sumScores = 0.0;
			for (int i = 0; i < int(vars.size()); ++i)
			{
				assert( vars[i].score(true) >=
					vars[i].score(false));
				sumScores += vars[i].score(true);
			}
			scores.clear();
			scores.push_back(sumScores);
			bestScore = -std::numeric_limits<double>::max();
			assert(sumScores > bestScore);

			omptl::sort(varConfidences.begin(),
				    varConfidences.end());

			const bool ok = solve();
			assert(ok);
			codeword = solution;
#ifndef NDEBUG
//std::cout << "Total nodes: " << nodes << " out of " <<
//	std::pow(2.0, double(NVariables)) << " likelihood " << 
//	std::exp(bestScore) << std::endl;
#endif
			return ok;
		}

		void link(const std::size_t variable, const std::size_t check)
		{
			assert(variable < NVariables);
			assert(check < NChecks);
			vars[variable].addLink(check);
			chks[check].addLink(variable);
		}

		void print()
		{
			assert(chks.size() == NChecks);
			for (std::size_t i = 0; i < chks.size(); ++i)
				std::cout << i << ": " <<
			cvmlcpp::to_string(chks[i].begin(), chks[i].end())
				<< std::endl;
		}

	private:
		bool consistent()
		{
			// Nr of set vars equal to depth of tree ?
			assert(NVariables-freeVars.count() == depth.back());
			assert(vids.size() == depth.back());

			// Correct bookkeeping for scores ?
			assert(scores.size() == depth.size());
			double sumScores = 0.0;
			for (std::size_t i = 0; i < vars.size(); ++i)
				sumScores += vars[i].score();
			assert(std::abs(1.0 - scores.back()/sumScores) < 0.001);

			// Are free vars really free ?
			for (std::size_t i = 0; i < vars.size(); ++i)
				assert(freeVars[i] == !vars[i].isSet());

			// Vars are either free or in the list of fixed vars
			for (std::size_t i = 0; i < vars.size(); ++i)
				assert( freeVars[i] ||
					(std::find(vids.begin(), vids.end(), i)
						!= vids.end()) );

			for (std::size_t i = 0; i < chks.size(); ++i)
			{
				// Nr Free vars ok ?
				std::size_t nFree = chks[i].size();
				for (MLCheck_::const_iterator
				     ci = chks[i].begin(); ci != chks[i].end();
				     ++ci)
					nFree -= freeVars[*ci] ? 0 : 1;
				assert(nFree == chks[i].nrFreeVars());

				// consistent state ?
				assert(!((chks[i].nrFreeVars() == 0u) &&
					(!chks[i].satisfied(vars))));
			}

			return true;
		}

		bool satisfied()
		{
			assert(consistent());
			for (std::size_t i = 0; i < chks.size(); ++i)
				if (!chks[i].satisfied(vars))
					return false;

			return true;
		}

	// constraintLength is nr of free vars is paritychecks under
	// consideration. Length of 0 indicates all possibles lengths.
	std::size_t selectByHeuristic(const std::size_t constraintLength)
	{
		using namespace std;

		tr1::array<std::tr1::array<std::size_t, NVariables>, 2u> q;

		fill(q[0].begin(), q[0].end(), 0u);
		fill(q[1].begin(), q[1].end(), 0u);

		for (std::size_t i = 0; i < NChecks; ++i)
			if ( (constraintLength == 0) ||
			     (constraintLength == chks[i].nrFreeVars()) )
			{
				// Count positive and negative occurences
				const std::size_t chkOk = chks[i].satisfied(vars);
				assert(chkOk <= 1u);
				for (MLCheck_::const_iterator v=chks[i].begin();
				     v != chks[i].end(); ++v)
					++q[chkOk][*v];
			}

		std::size_t winner = NVariables;
		double minQScore = std::numeric_limits<double>::max();
		for (std::size_t v = 0; v < NVariables; ++v)
			if (freeVars[v])
			{
// 				if (std::min(q[0][v], q[1][v]) > 0)
// 				{
					const double d = 0.0001;
					const double qScore =
						  (d + abs(vars[v].p()-0.5))
						//----------------------------
						/ (d+double(q[0][v] * q[1][v]));

					if (qScore < minQScore)
					{
						minQScore = qScore;
						winner   = v;
					}
// 				}
			}

		assert((winner == NVariables) || freeVars[winner]);
		return winner;
	}

std::size_t selectNextBranchVariable()
{
	assert(consistent());

	// Consider 2-var constraints; 3-var constraints, of all constraints
	std::size_t branchvar = selectByHeuristic(2) ;
	if ( branchvar == NVariables)
		if ( (branchvar = selectByHeuristic(3)) == NVariables)
			branchvar = selectByHeuristic(0);
	assert(branchvar != NVariables);
	assert(consistent());

	return branchvar;
}

		bool setPropagatedVariable(const std::size_t v, bool mostLikely)
		{
			assert(freeVars[v]);
			vars[v].set(mostLikely);
			typedef MLVariable_::const_iterator ci;
			for (ci c = vars[v].begin(); c != vars[v].end(); ++c)
			{
				// Conflict ?
				if ( ( chks[*c].setVar(v) == 0u) &&
				     (!chks[*c].satisfied(vars)) )
				{
// for (std::size_t i = 0; i < depth.back(); ++i)
// 	std::cout << "   ";
// std::cout <<"setPropagatedVariable() STOP: Conflict in chk "<<*c<<std::endl;

					// Roll-back
					for (ci j = vars[v].begin(); j != c;++j)
						chks[*j].unsetVar(v);
					chks[*c].unsetVar(v);
					vars[v].unset();
					return false;
				}
			}

			freeVars[v] = false;
			vids.push_back(v);

			return true;
		}

		bool setVariable(const std::size_t v, bool mostLikely)
		{
			assert(consistent());
			if (!setPropagatedVariable(v, mostLikely))
				return false;

			depth.push_back(depth.back() + 1);

			const double newScore = scores.back() +
				( (vars[v].value() == vars[v].mlValue()) ? 0.0 :
				  (vars[v].score(false)-vars[v].score(true)) );

			scores.push_back(newScore);

			assert(consistent());

			// Conflict ?
			if (!propagate(v)) // Prop
			{
				// Roll-back
				depth.pop_back();
				scores.pop_back();
				unSetPropagatedVariable(v);
				assert(consistent());
				return false;
			}

			assert(consistent());
			return true;
		}


		void unSetVariable(const std::size_t v)
		{
			assert(consistent());
			assert(!freeVars[v]);

			// Undo Propagation
 			dePropagate();

			assert(consistent());

			// Unset variable itself
			unSetPropagatedVariable(v);
			depth.pop_back();
			scores.pop_back();

			assert(consistent());
		}

		void unSetPropagatedVariable(const std::size_t v)
		{
			typedef MLVariable_::const_iterator ci;
			for (ci i = vars[v].begin(); i != vars[v].end(); ++i)
				chks[*i].unsetVar(v);

			vars[v].unset();

			vids.pop_back();
			freeVars[v] = true;
		}

		void dePropagate()
		{
			assert(consistent());
			std::size_t d = depth.back();
			depth.pop_back();
			scores.pop_back();
			assert(depth.back() > 0u);
			while (--d >= depth.back())
				unSetPropagatedVariable(vids.back());
			assert(consistent());
		}

		typedef std::pair<std::size_t, bool> VarVal;

bool propagate(const std::size_t v)
{
	assert(consistent());
	typedef MLVariable_::const_iterator cvi;
	typedef MLCheck_::const_iterator cci;

	assert(vids.back() == v);
	stack.clear();
	for (cvi chk = vars[v].begin(); chk != vars[v].end(); ++chk)
		if (chks[*chk].nrFreeVars() == 1u) // propagate!
		{
			// Find last free var in this parity chk
			std::size_t lastVar = -1;
			for (cci j=chks[*chk].begin(); j!=chks[*chk].end(); ++j)
				if (freeVars[*j])
				{
					lastVar = *j;
					break;
				}
			assert(lastVar < NVariables);

			// Push that var + the value it must have
			bool mostLikely = chks[*chk].satisfied(vars);
			if (vars[lastVar].value() != vars[lastVar].mlValue())
				mostLikely = !mostLikely;
			stack.push_back(VarVal(lastVar, mostLikely));
		}

	std::size_t d = 0u;
	double score = scores.back();
	while (stack.size() > 0u)
	{
		const VarVal vs = stack.back();
		stack.pop_back();
// std::cout << (depth.back() + d) << " " << vids.size() << std::endl;
		assert(depth.back() + d == vids.size());
		assert(NVariables-freeVars.count() == depth.back() + d);

		const std::size_t varID = vs.first;
		if (vars[varID].isSet()) // already propagated ?
			continue;

		const bool mostLikely = vs.second;
		const double newScore = score +
			( mostLikely ? 0.0 :
			 (vars[varID].score(false)-vars[varID].score(true)) );

// for (std::size_t i = 0; i < depth.back(); ++i)
// 	std::cout << "   ";
// std::cout << "Propagate() Force var: " << varID << " to ml: " << mostLikely
// 	<< " value: " << vars[varID].value(mostLikely) << std::endl;

		if ( (newScore > bestScore) &&
		      setPropagatedVariable(varID, mostLikely) )
		{
			++d;
			score = newScore;
			for (cvi c = vars[v].begin(); c != vars[v].end(); ++c)
				assert( (chks[*c].nrFreeVars() > 0) ||
					 chks[*c].satisfied(vars) );
		}
		else
		{
			// Roll-back
			depth.push_back(depth.back() + d); // push to complete
			scores.push_back(score);	   // propagation...
			assert(consistent());
			dePropagate();			   // ... then undo

// for (std::size_t i = 0; i < depth.back(); ++i)
// 	std::cout << "   ";
// if (!(newScore > bestScore))
// 	std::cout << "Propagate() STOP propagation var " << vs.first 
// 		<< " setting " << v << " score: " << newScore << " !> " 
// 		<< bestScore << std::endl;
// else
// 	std::cout<<"Propagate(): STOP propagation var " << vs.first
// 		<< " setting " << v << " CONFLICT" << std::endl;

			assert(consistent());
			return false;
		}

		for (cvi c = vars[vs.first].begin();c!=vars[vs.first].end();++c)
			if (chks[*c].nrFreeVars() == 1u) // propagate!
			{
				std::size_t lastVar = NVariables; // Invalid value
				for (cci j = chks[*c].begin();
				     j != chks[*c].end(); ++j)
					if (freeVars[*j])
					{
						lastVar = *j;
						break;
					}
				assert(lastVar < NVariables);

				bool mostLikely = chks[*c].satisfied(vars);
				if (vars[lastVar].value() !=
				    vars[lastVar].mlValue())
					mostLikely = !mostLikely;
				stack.push_back(VarVal(lastVar, mostLikely));
			}
	}
	depth.push_back(depth.back() + d);
	scores.push_back(score);
	assert(consistent());
	return true;
}

bool solve()
{
	assert(depth.size() > 0);
	assert(consistent());
	assert(!satisfied());

#ifndef NDEBUG
	++nodes;
/*	if (nodes % (1u<<16u) == 0)
		std::cout << "Nodes: " << nodes << " depth:" << depth.back()
			<< std::endl;*/
#endif

	assert(scores.size() > 0u);
	const std::size_t v = selectNextBranchVariable();
	assert(v < NVariables);
	assert(freeVars[v]);
// std::cout << "vars set: "<<vids.size()<<" depth: "<<depth.back()<<std::endl;
	assert(vids.size() >= depth.back());

	bool solved = false;

	for (int ml = 0; ml <= 1 /* && !solved */; ++ml)
	{
		const bool mostLikely = static_cast<bool>(ml);

		assert(consistent());

		// Adjust score if we don't use the most likely assignment
		// for this variable.
		const double score = scores.back() +
			( mostLikely ? 0.0 :
				(vars[v].score(false)-vars[v].score(true)) );

// for (std::size_t i = 0; i < depth.back(); ++i)
// 	std::cout << "   ";
// std::cout << "solve() Depth: " << depth.back() << " Branch: " << v
// 	<< " p: " << vars[v].p() << " trying ML:" << mostLikely << " value: "
// 	<< vars[v].value(mostLikely) << " score: " << score << std::endl;

		if (!(score > bestScore))
		{
// for (std::size_t i = 0; i < depth.back(); ++i)
// 	std::cout << "   ";
// std::cout << "solve() STOP: Score: " << scores.back() << " + "
// 	<< ( mostLikely ? 0.0 : (vars[v].score(false)-vars[v].score(true)) )
// 	<< " = " << score << " <= " << bestScore << std::endl;

// 			assert( (depth.back() > 1u) || (ml == 0u) || solved );
			continue;
		}
		assert(score > bestScore);

		if (setVariable(v, mostLikely))
		{
			assert(consistent());
			if (satisfied())
			{
//  << solution.to_string() << std::endl;
// for (std::size_t i = 0; i < depth.back(); ++i)
// 	std::cout << "   ";
// std::cout << "Solution Found. Score: " << score << " vs. " << bestScore
// 	<< std::endl;

				if (score > bestScore)
				{
					bestScore = score;
					for (std::size_t i = 0; i<vars.size(); ++i)
						solution.set(i,vars[i].value());
// for (std::size_t i = 0; i < depth.back(); ++i)
// 	std::cout << "   ";
// std::cout << "Solution Accepted! Score: " << std::exp(bestScore)
// 	<< std::endl;

					solved = true;
				}
			}
			else if (freeVars.count() > 0u)
				solved = solve() || solved;
			else
			{
				for (std::size_t i = 0; i < depth.back(); ++i)
					std::cout << "   ";
				std::cout << "solve() STOP: no more free vars!"
					<< std::endl;
			}

			assert(consistent());
			unSetVariable(v);
			assert(consistent());
		}
// 		else
// 			assert( (depth.back() > 1u) || (ml == 0u) || solved );
		assert(freeVars[v]);

		assert(consistent());
	}

	return solved;
}

		double						bestScore;
		std::bitset<NVariables>				solution;
		std::bitset<NVariables>				freeVars;
		std::vector<std::size_t>			vids;
		std::vector<double>				scores;
		std::vector<std::size_t>			depth;
		std::tr1::array<_VarConfidence, NVariables> varConfidences;
		std::tr1::array<MLVariable_, NVariables>	vars;
		std::tr1::array<MLCheck_,    NChecks>	chks;
		std::vector<VarVal> stack;
#ifndef NDEBUG
		std::size_t 					nodes;
#endif
};

} // namespace

#endif
