/*
 * Bone.cpp
 *
 *  Created on: 19 oct. 2013
 *
 */

#include "Bone.h"

#include "GL/glew.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>


#include "logger/Logger.h"

void Bone::calcMatrix(std::vector<Bone> & allbones, ssize_t anim, size_t time, bool rotate)
{
  if (calc)
    return;

  glm::mat4 m(1.0f);
  // The bone's own rotation. It used to be applied to the matrix without being kept, so the
  // normal matrix below was built from an identity: normals, light directions and particle
  // emission ignored every bone rotation -- an emitter turned to spray sideways sprayed
  // straight up.
  glm::fquat q;

  bool tr = rot.uses(anim) || scale.uses(anim) || trans.uses(anim) || billboard;
  if (tr) 
  {
    m = glm::translate(m, pivot);

    if (trans.uses(anim))
      m = glm::translate(m, trans.getValue(anim, time));

    if (rot.uses(anim) && rotate)
    {
      q = glm::normalize(rot.getValue(anim, time));
      m = m * glm::toMat4(q);
    }

    if (scale.uses(anim))
      m = glm::scale(m, scale.getValue(anim, time));

    m = glm::translate(m, pivot * -1.0f);
  }

  // Guard the upper bound too: an out-of-range parent (from a malformed/modern rig)
  // would otherwise make allbones[parent] reference - and calcMatrix() WRITE - memory
  // past the end of the bones[] vector every frame, corrupting the adjacent heap
  // allocation. Parents are normally sanitized at load (WoWModel::initAnimated); this
  // is defence-in-depth for any path that didn't go through it.
  const bool hasParent = parent > -1 && parent < (int)allbones.size();
  if (hasParent)
  {
    allbones[parent].calcMatrix(allbones, anim, time, rotate);
    mat = allbones[parent].mat * m;
  }
  else
  {
    mat = m;
  }

  if (billboard)
  {
    // Turned toward the camera after the parents are applied, not before: a billboard faces
    // the viewer whatever its parents do. The old code overwrote two rows of the local matrix
    // -- columns in the matrix library it was written for, rows in glm -- and in a Y-up space
    // this engine no longer uses: billboards tipped over, and flattened into a line seen from
    // the side. Only the spherical kind was handled; the cylindrical ones stood still.
    float modelview[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
    // The modelview's rows are the camera's axes in the model's space.
    const glm::mat3 camera = glm::transpose(glm::mat3(glm::make_mat4(modelview)));
    mat = billboardMatrix(billboard, mat, glm::mat3_cast(q), pivot, camera);

    mrot = glm::mat4(1.0f);
    for (int c = 0; c < 3; c++)
    {
      const glm::vec3 axis(mat[c]);
      const float l = glm::length(axis);
      mrot[c] = glm::vec4(l > 0.0f ? axis / l : axis, 0.0f);
    }
  }
  else
  {
    // A bone without a rotation of its own still turns with its parents.
    mrot = hasParent ? allbones[parent].mrot * glm::toMat4(q) : glm::toMat4(q);
  }

  transPivot = glm::vec3(mat * glm::vec4(pivot, 1.0f));

  calc = true;
}

void Bone::initV3(GameFile & f, ModelBoneDef &b, const modelAnimData & data)
{
  calc = false;

  parent = b.parent;
  pivot = b.pivot;
  billboard = uint32(b.flags) & MODELBONE_BILLBOARD_ANY;

  boneDef = b;

  trans.init(b.translation, f, data);
  rot.init(b.rotation, f, data);
  scale.init(b.scaling, f, data);
}

