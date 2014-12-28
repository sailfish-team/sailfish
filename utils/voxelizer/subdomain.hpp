// Utility functions for dealing with Sailfish geometry data.
#ifndef SLF_SUBDOMAIN_H
#define SLF_SUBDOMAIN_H 1

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/math/Euclid>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>
#include <cvmlcpp/volume/VoxelTools>

using namespace cvmlcpp;

typedef DTree<char, 3u> Octree;

// Returns the location (origin) of a node.
iPoint3D NodeLocation(const Octree::DNode& node, const int max_depth);

// Returns the location of the point opposite to the origin.
iPoint3D NodeExtent(const Octree::DNode& node, const int max_depth);

// Returns the number of fluid voxels corresponding to an octree node.
int CountFluidNodes(const Octree::DNode& node, const int max_depth);

// Clears an internal cache mapping nodes to the number of fluid voxels
// they contain.
void FlushFluidCache();

// Removes all children nodes that do no contain any fluid.
void RemoveEmptyAreas(Octree::DNode node);

extern const char kFluid;
extern const char kWall;

// Represents a cuboid subdomain.
// Tracks fraction of active (fluid) nodes.
class Subdomain {
 public:
  Subdomain(iPoint3D origin, iPoint3D extent):
      origin_(origin), extent_(extent),
      fluid_nodes_(0) {};

  Subdomain(iPoint3D origin, iPoint3D extent, int fluid_nodes):
      origin_(origin), extent_(extent),
      fluid_nodes_(fluid_nodes) {};

  Subdomain(const Octree::DNode& node, int max_depth):
    origin_(NodeLocation(node, max_depth)), extent_(NodeExtent(node, max_depth)),
    fluid_nodes_(CountFluidNodes(node, max_depth)) {};

  bool operator==(const Subdomain& rhs) const {
    return this->origin_ == rhs.origin_ &&
      this->extent_ == rhs.extent_ &&
      this->fluid_nodes_ == rhs.fluid_nodes_;
  }

  std::string JSON() const {
    std::ostringstream c;
    c << "{ pos: [" << origin_.x() << ", "
            << origin_.y() << ", "
            << origin_.z() << "], "
      << "size: [" << extent_.x() - origin_.x() + 1 << ", "
                   << extent_.y() - origin_.y() + 1 << ", "
             << extent_.z() - origin_.z() + 1 << "] }";
    return c.str();
  }

  // Builds the union of two subdomains.
  const Subdomain operator+(const Subdomain& rhs) const {
    Subdomain result = *this;
    result.origin_.set(
        std::min(result.origin_.x(), rhs.origin_.x()),
        std::min(result.origin_.y(), rhs.origin_.y()),
        std::min(result.origin_.z(), rhs.origin_.z()));
    result.extent_.set(
        std::max(result.extent_.x(), rhs.extent_.x()),
        std::max(result.extent_.y(), rhs.extent_.y()),
        std::max(result.extent_.z(), rhs.extent_.z()));
    result.fluid_nodes_ += rhs.fluid_nodes_;
    return result;
  }

  bool contains(const Subdomain& other) const {
    return
      other.origin_.x() >= this->origin_.x() &&
      other.extent_.x() <= this->extent_.x() &&
      other.origin_.y() >= this->origin_.y() &&
      other.extent_.y() <= this->extent_.y() &&
      other.origin_.z() >= this->origin_.z() &&
      other.extent_.z() <= this->extent_.z();
  }

  int len() const {
    return (extent_.x() - origin_.x() + 1);
  }

  // Returns the number of nodes contained within the subdomain.
  int volume() const {
    return (extent_.x() - origin_.x() + 1) *
         (extent_.y() - origin_.y() + 1) *
         (extent_.z() - origin_.z() + 1);
  }

  int fluid_nodes() const {
    return fluid_nodes_;
  }

  void add_fluid(int fluid) {
    fluid_nodes_ += fluid;
  }

  double fill_fraction() const {
    return static_cast<double>(fluid_nodes_) / volume();
  }

  friend std::ostream& operator<<(std::ostream& os, const Subdomain& s);

  private:
  iPoint3D origin_, extent_;  // location of the origin point and the point
                              // opposite to the origin
  int fluid_nodes_;     // number of fluid nodes in the subdomain
};

std::vector<Subdomain> ToSubdomains(const Octree::DNode node, const int max_depth);

#endif  // SLF_SUBDOMAIN_H
