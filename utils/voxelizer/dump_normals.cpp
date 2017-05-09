// Prints normal vectors for every facet to stdout.
//
// The output format is:
//   <facet centroid> <facet normal>

#include <cstdio>
#include <cvmlcpp/math/Euclid>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <typeinfo>

int main(int argc, char **argv) {
  cvmlcpp::Geometry<float> geometry;

  if (argc < 1) {
    std::cerr << "Usage: ./dump_normals <STL file>" << std::endl << std::endl;
    return -1;
  }

  readSTL(geometry, argv[1]);
  for (auto facet_it = geometry.facetsBegin();
       facet_it != geometry.facetsEnd(); ++facet_it) {
    cvmlcpp::fPoint3D center(0.0f, 0.0f, 0.0f);
    for (const size_t key : *facet_it) center += geometry.point(key);
    const auto normal = facet_it->normal();
    std::cout
      << center.x() / 3.0 << " "
      << center.y() / 3.0 << " "
      << center.z() / 3.0 << " "
      << normal.x() << " "
      << normal.y() << " "
      << normal.z() << std::endl;
  }
  return 0;
}
