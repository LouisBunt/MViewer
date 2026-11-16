/*
 * Bone.h
 *
 *  Created on: 19 oct. 2013
 *
 */

#ifndef _BONE_H_
#define _BONE_H_

#include "animated.h"
#include "modelheaders.h" // ModelBoneDef

#include "glm/glm.hpp"

class GameFile;
class WoWModel;



class Bone {
public:
  Animated<glm::vec3> trans;
  Animated<glm::fquat, PACK_QUATERNION, Quat16ToQuat32> rot;
  Animated<glm::vec3> scale;

  glm::vec3 pivot, transPivot;
  int16 parent;

  // The bone's MODELBONE_BILLBOARD* flags: how it turns toward the camera. 0 = it does not.
  uint32 billboard;
  glm::mat4 mat;
  // The rotation in `mat` alone, for what turns but does not move: normals, light directions,
  // the direction particles leave an emitter in.
  glm::mat4 mrot;

  ModelBoneDef boneDef;

  bool calc;
  void calcMatrix(std::vector<Bone> & allbones, ssize_t anim, size_t time, bool rotate=true);
  void initV3(GameFile & f, ModelBoneDef &b, const modelAnimData & data);

  // A billboard bone's matrix. `full` is the bone's matrix as animated (parent * own), `own`
  // the rotation of its own animation alone, and `camera` the camera's axes in the model's
  // space: right, up, and back toward the viewer, as columns. Returns `full` turned to face
  // the camera, with its scale, and with the pivot where `full` puts it.
  //
  // WoW authors a billboard facing +X: in the model's front view (camera on +X, Z up) it is
  // drawn as built, so there the result is `full` unchanged.
  //  - spherical (MODELBONE_BILLBOARD): X toward the viewer, Y to the right, Z up -- whatever
  //    the parents do -- and the bone's own rotation on top, so a spinning glow still spins;
  //  - cylindrical: the locked axis keeps the direction `full` gives it, and the bone turns
  //    about it until the axis next to it lies flat across the screen.
  static glm::mat4 billboardMatrix(uint32 flags, const glm::mat4 & full, const glm::mat3 & own,
                                   const glm::vec3 & pivot, const glm::mat3 & camera)
  {
    const float tiny = 1e-6f;
    glm::vec3 x(full[0]), y(full[1]), z(full[2]);
    const glm::vec3 scale(glm::length(x), glm::length(y), glm::length(z));
    const glm::vec3 right = unit(camera[0]), up = unit(camera[1]), back = unit(camera[2]);

    if (flags & MODELBONE_BILLBOARD)
    {
      const glm::mat3 facing = glm::mat3(back, right, up) * own;
      x = facing[0];
      y = facing[1];
      z = facing[2];
    }
    else if (flags & (MODELBONE_BILLBOARD_LOCK_X | MODELBONE_BILLBOARD_LOCK_Y | MODELBONE_BILLBOARD_LOCK_Z))
    {
      // Looking straight along the locked axis no turn about it helps: leave the bone as is.
      if (flags & MODELBONE_BILLBOARD_LOCK_X)
      {
        x = unit(x);
        const glm::vec3 across = glm::cross(x, back);
        if (glm::length(across) < tiny)
          return full;
        y = unit(across);
        z = glm::cross(x, y);
      }
      else if (flags & MODELBONE_BILLBOARD_LOCK_Y)
      {
        y = unit(y);
        const glm::vec3 across = glm::cross(back, y);
        if (glm::length(across) < tiny)
          return full;
        x = unit(across);
        z = glm::cross(x, y);
      }
      else
      {
        z = unit(z);
        const glm::vec3 across = glm::cross(z, back);
        if (glm::length(across) < tiny)
          return full;
        y = unit(across);
        x = glm::cross(y, z);
      }
    }
    else
      return full;

    glm::mat4 m(1.0f);
    m[0] = glm::vec4(x * scale.x, 0.0f);
    m[1] = glm::vec4(y * scale.y, 0.0f);
    m[2] = glm::vec4(z * scale.z, 0.0f);
    const glm::vec3 at(full * glm::vec4(pivot, 1.0f));
    m[3] = glm::vec4(at - glm::mat3(m) * pivot, 1.0f);
    return m;
  }

private:
  static glm::vec3 unit(const glm::vec3 & v)
  {
    const float l = glm::length(v);
    return l > 0.0f ? v / l : v;
  }
};


#endif /* _BONE_H_ */
