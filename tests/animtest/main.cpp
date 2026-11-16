// Tests for the animation tracks (engine/Source/games/wow/animated.h) and the bones' billboards
// (Bone.h): which buffer each sequence's keys are read from, the keyframe search, global
// sequences, aliases, and how a billboard turns toward the camera. Every case is a way a real
// model's animation used to come apart -- a bone flung to infinity, a part collapsed onto its
// pivot, an effect frozen on its first frame, a glow tipped over or flattened into a line.
//
// The tracks are fed hand-built byte images laid out the way an M2 and its .anim files store
// them, through Animated::load(), the same code both init()s run. No game data, no engine.
//
//   animtest
//
// Exit code = number of failing checks, so a plain shell/CI check works.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

#include "glm/gtc/matrix_transform.hpp"

#include "Bone.h"
#include "animated.h"

namespace {

int failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
      failures++;                                                                \
    }                                                                            \
  } while (0)

bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }
bool approx(const glm::vec3 & a, const glm::vec3 & b, float eps = 1e-4f)
{
  return approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps);
}
bool approx(const glm::mat4 & a, const glm::mat4 & b)
{
  for (int c = 0; c < 4; c++)
    for (int r = 0; r < 4; r++)
      if (!approx(a[c][r], b[c][r], 1e-4f))
        return false;
  return true;
}

// A byte image: an M2's MD21 chunk, or an .anim file.
struct Blob
{
  std::vector<unsigned char> bytes;

  template <class T>
  uint32 put(const T & v)
  {
    const uint32 at = uint32(bytes.size());
    const unsigned char * p = reinterpret_cast<const unsigned char *>(&v);
    bytes.insert(bytes.end(), p, p + sizeof(T));
    return at;
  }

  template <class T>
  uint32 putArray(const std::vector<T> & v)
  {
    const uint32 at = uint32(bytes.size());
    for (const T & x : v)
      put(x);
    return at;
  }

  template <class T>
  void patch(uint32 at, const T & v) { std::memcpy(&bytes[at], &v, sizeof(T)); }
};

// One sequence of a track as the files store it. Hermite and Bezier keys are three values
// each: the value and its two tangents.
template <class D>
struct Seq
{
  std::vector<uint32> times;
  std::vector<D> keys;
  Blob * home = nullptr;   // the file the arrays are in; nullptr = the model itself
};

// Writes the track's per-sequence array headers into `model` and each sequence's arrays into
// its home file -- offsets relative to that file, as for a sequence in an .anim file.
template <class D>
AnimationBlock writeTrack(Blob & model, int16 type, int16 globalSeq, const std::vector<Seq<D> > & seqs)
{
  const uint32 n = uint32(seqs.size());
  const uint32 perKey = (type == INTERPOLATION_HERMITE || type == INTERPOLATION_BEZIER) ? 3 : 1;
  const AnimationBlockHeader none = { 0, 0 };
  const uint32 ofsTimes = uint32(model.bytes.size());
  for (uint32 j = 0; j < n; j++)
    model.put(none);
  const uint32 ofsKeys = uint32(model.bytes.size());
  for (uint32 j = 0; j < n; j++)
    model.put(none);
  for (uint32 j = 0; j < n; j++)
  {
    Blob & home = seqs[j].home ? *seqs[j].home : model;
    const AnimationBlockHeader t = { uint32(seqs[j].times.size()), home.putArray(seqs[j].times) };
    const AnimationBlockHeader k = { uint32(seqs[j].keys.size() / perKey), home.putArray(seqs[j].keys) };
    model.patch(ofsTimes + j * uint32(sizeof(AnimationBlockHeader)), t);
    model.patch(ofsKeys + j * uint32(sizeof(AnimationBlockHeader)), k);
  }
  AnimationBlock b;
  b.type = type;
  b.seq = globalSeq;
  b.nTimes = n;
  b.ofsTimes = ofsTimes;
  b.nKeys = n;
  b.ofsKeys = ofsKeys;
  return b;
}

// What modelAnimData.animfiles says, for the plain buffers here: a sequence absent from the map
// has its keys in the model, one mapped to nullptr in an .anim file that could not be found.
struct AnimFiles
{
  std::map<size_t, const Blob *> files;

  bool operator()(size_t j, const unsigned char *& buf, size_t & size) const
  {
    auto it = files.find(j);
    if (it == files.end())
      return true;
    if (!it->second)
      return false;
    buf = it->second->bytes.data();
    size = it->second->bytes.size();
    return true;
  }
};

template <class T, class D, class C>
void load(Animated<T, D, C> & track, const AnimationBlock & b, const Blob & model,
          const AnimFiles & files = AnimFiles(), const std::vector<uint32> & globals = {})
{
  track.load(b, globals, model.bytes.data(), model.bytes.size(), files);
}

const glm::vec3 V(float v) { return glm::vec3(v, v, v); }

// --- the keyframe search ---------------------------------------------------------------

void testFindSegment()
{
  const std::vector<size_t> t = { 10, 20, 30 };
  animtrack::Segment s;

  // Before the first key the first key holds. The old search computed time - t1 in size_t
  // here, which underflowed to a ratio near 1.8e19.
  s = animtrack::findSegment(t, 3, 0);
  CHECK(s.index == 0 && s.ratio == 0.0f);
  s = animtrack::findSegment(t, 3, 10);
  CHECK(s.index == 0 && s.ratio == 0.0f);

  s = animtrack::findSegment(t, 3, 15);
  CHECK(s.index == 0 && approx(s.ratio, 0.5f));
  s = animtrack::findSegment(t, 3, 20);
  CHECK(s.index == 1 && s.ratio == 0.0f);
  s = animtrack::findSegment(t, 3, 25);
  CHECK(s.index == 1 && approx(s.ratio, 0.5f));

  // Exactly on the last key and past it: the last key, not an extrapolated first segment.
  s = animtrack::findSegment(t, 3, 30);
  CHECK(s.index == 2 && s.ratio == 0.0f);
  s = animtrack::findSegment(t, 3, 100000);
  CHECK(s.index == 2 && s.ratio == 0.0f);

  // Only the first `count` times count: the arrays can be longer than the keys.
  s = animtrack::findSegment(t, 2, 25);
  CHECK(s.index == 1 && s.ratio == 0.0f);

  // A single key, or none to interpolate between.
  s = animtrack::findSegment(t, 1, 25);
  CHECK(s.index == 0 && s.ratio == 0.0f);

  // Two keys on the same time used to divide by zero. Every ratio stays in 0..1.
  const std::vector<size_t> dup = { 0, 10, 10, 10, 20 };
  bool bounded = true;
  for (size_t time = 0; time <= 25; time++)
  {
    s = animtrack::findSegment(dup, dup.size(), time);
    bounded = bounded && std::isfinite(s.ratio) && s.ratio >= 0.0f && s.ratio <= 1.0f
              && s.index < dup.size();
  }
  CHECK(bounded);
  s = animtrack::findSegment(dup, dup.size(), 15);
  CHECK(s.index == 3 && approx(s.ratio, 0.5f));

  // Keys out of order (a malformed file) give some segment, never one past the end.
  const std::vector<size_t> unsorted = { 0, 50, 20, 30, 100 };
  bounded = true;
  for (size_t time = 0; time <= 110; time += 5)
  {
    s = animtrack::findSegment(unsorted, unsorted.size(), time);
    bounded = bounded && std::isfinite(s.ratio) && s.ratio >= 0.0f && s.ratio <= 1.0f
              && s.index < unsorted.size();
  }
  CHECK(bounded);
}

// --- global sequences ------------------------------------------------------------------

void testTrackTime()
{
  const std::vector<uint32> globals = { 0, 1000 };

  // Not on a global sequence: the time within the playing sequence.
  CHECK(animtrack::trackTime(-1, globals, 123, 99999) == 123);

  // On one: the model-wide clock, wrapped to the sequence's length.
  CHECK(animtrack::trackTime(1, globals, 123, 2500) == 500);

  // A global sequence of length 0 holds its first key, as does an index past the list.
  CHECK(animtrack::trackTime(0, globals, 123, 2500) == 0);
  CHECK(animtrack::trackTime(7, globals, 123, 2500) == 0);
  CHECK(animtrack::trackTime(0, std::vector<uint32>(), 123, 2500) == 0);
}

void testGlobalTrack()
{
  // A track on global sequence 1 (200 ms): one array, read from the model -- never from an
  // .anim file, even when sequence 0 has one, missing or not.
  Blob model;
  std::vector<Seq<glm::vec3> > seqs(1);
  seqs[0].times = { 0, 100, 200 };
  seqs[0].keys = { V(0), V(10), V(0) };
  const AnimationBlock b = writeTrack(model, INTERPOLATION_LINEAR, 1, seqs);

  AnimFiles missing;
  missing.files[0] = nullptr;
  Animated<glm::vec3> track;
  load(track, b, model, missing, { 0, 200 });

  CHECK(track.uses(0));
  CHECK(track.uses(57));        // whatever sequence plays
  globalTime = 50;
  CHECK(approx(track.getValue(0, 0), V(5)));
  globalTime = 250;             // wraps at 200
  CHECK(approx(track.getValue(3, 0), V(5)));
  CHECK(approx(track.getValue(3, 999), V(5)));   // the sequence's own time does not matter
  globalTime = 0;
}

void testStillGlobalHoldsFirstKey()
{
  // A global sequence of length 0 -- common on item effects that do not move. It used to
  // return a default value: a zero vector, which on a scale track collapses every vertex of
  // the bone onto its pivot.
  Blob model;
  std::vector<Seq<glm::vec3> > seqs(1);
  seqs[0].times = { 0, 100 };
  seqs[0].keys = { V(1), V(3) };
  const AnimationBlock b = writeTrack(model, INTERPOLATION_LINEAR, 0, seqs);

  Animated<glm::vec3> scale;
  load(scale, b, model, AnimFiles(), { 0 });
  globalTime = 12345;
  CHECK(approx(scale.getValue(0, 50), V(1)));

  // Same for an index past the model's list of global sequences.
  Blob model2;
  const AnimationBlock b2 = writeTrack(model2, INTERPOLATION_LINEAR, 4, seqs);
  Animated<glm::vec3> scale2;
  load(scale2, b2, model2, AnimFiles(), { 0 });
  CHECK(approx(scale2.getValue(0, 50), V(1)));
  globalTime = 0;
}

// --- where each sequence's keys are ----------------------------------------------------

void testKeysPerSequence()
{
  // Sequence 0 in the model; 1 and 2 -- two variations of one animation, say Stand 0 and
  // Stand 1 -- each in its own .anim file, at different offsets. Each must read its own file.
  // They used to be looked up by animation id, so every variation read the file registered
  // last, and the model's own sequence read a sibling's.
  Blob model, anim1, anim2;
  anim2.put(uint32(0xDEADBEEF));   // shift sequence 2's arrays away from sequence 1's offsets
  anim2.put(uint32(0xDEADBEEF));
  std::vector<Seq<glm::vec3> > seqs(3);
  seqs[0].times = { 0, 100 };
  seqs[0].keys = { V(1), V(2) };
  seqs[1].times = { 0, 100 };
  seqs[1].keys = { V(5), V(6) };
  seqs[1].home = &anim1;
  seqs[2].times = { 0, 50, 100 };
  seqs[2].keys = { V(7), V(8), V(9) };
  seqs[2].home = &anim2;
  const AnimationBlock b = writeTrack(model, INTERPOLATION_LINEAR, -1, seqs);

  AnimFiles files;
  files.files[1] = &anim1;
  files.files[2] = &anim2;
  Animated<glm::vec3> trans;
  load(trans, b, model, files);

  CHECK(trans.uses(0) && trans.uses(1) && trans.uses(2));
  CHECK(approx(trans.getValue(0, 50), V(1.5f)));
  CHECK(approx(trans.getValue(1, 50), V(5.5f)));
  CHECK(approx(trans.getValue(2, 25), V(7.5f)));
  CHECK(approx(trans.getValue(2, 75), V(8.5f)));
}

void testMissingAnimFile()
{
  // Sequence 1's keys are in an .anim file that is not there. The sequence has no keys -- the
  // bone keeps its rest pose -- instead of reading the model at offsets meant for the file.
  Blob model, anim1;
  std::vector<Seq<glm::vec3> > seqs(2);
  seqs[0].times = { 0, 100 };
  seqs[0].keys = { V(1), V(2) };
  seqs[1].times = { 0, 100 };
  seqs[1].keys = { V(5), V(6) };
  seqs[1].home = &anim1;
  const AnimationBlock b = writeTrack(model, INTERPOLATION_LINEAR, -1, seqs);

  AnimFiles files;
  files.files[1] = nullptr;
  Animated<glm::vec3> trans;
  load(trans, b, model, files);

  CHECK(trans.uses(0));
  CHECK(!trans.uses(1));
  CHECK(approx(trans.getValue(0, 100), V(2)));
}

void testManySequences()
{
  // Modern characters have more than 500 sequences; the arrays used to stop at 500.
  Blob model;
  std::vector<Seq<float> > seqs(600);
  for (size_t j = 0; j < seqs.size(); j++)
  {
    seqs[j].times = { 0, 100 };
    seqs[j].keys = { float(j), float(j) + 1.0f };
  }
  const AnimationBlock b = writeTrack(model, INTERPOLATION_LINEAR, -1, seqs);
  Animated<float> track;
  load(track, b, model);
  CHECK(track.uses(599));
  CHECK(approx(track.getValue(599, 50), 599.5f));
  CHECK(approx(track.getValue(500, 0), 500.0f));
  CHECK(!track.uses(600));
  CHECK(!track.uses(-1));
}

// --- what a track does with odd data ---------------------------------------------------

void testEnds()
{
  Blob model;
  std::vector<Seq<float> > seqs(1);
  seqs[0].times = { 100, 200 };
  seqs[0].keys = { 1.0f, 3.0f };
  const AnimationBlock b = writeTrack(model, INTERPOLATION_LINEAR, -1, seqs);
  Animated<float> track;
  load(track, b, model);

  CHECK(approx(track.getValue(0, 0), 1.0f));       // before the first key
  CHECK(approx(track.getValue(0, 150), 2.0f));
  CHECK(approx(track.getValue(0, 200), 3.0f));     // on the last key
  CHECK(approx(track.getValue(0, 99999), 3.0f));   // past it

  // No interpolation: each key holds until the next.
  Blob model2;
  const AnimationBlock b2 = writeTrack(model2, INTERPOLATION_NONE, -1, seqs);
  Animated<float> step;
  load(step, b2, model2);
  CHECK(approx(step.getValue(0, 199), 1.0f));
  CHECK(approx(step.getValue(0, 200), 3.0f));
}

void testCountsClampedToFile()
{
  // Array headers that claim more entries than the file holds -- a truncated file, or a
  // format change -- are read up to the end of the buffer, never past it.
  Blob model;
  std::vector<Seq<float> > seqs(1);
  seqs[0].times = { 0, 100 };
  seqs[0].keys = { 1.0f, 3.0f };
  const AnimationBlock b = writeTrack(model, INTERPOLATION_LINEAR, -1, seqs);
  AnimationBlockHeader t, k;
  std::memcpy(&t, &model.bytes[b.ofsTimes], sizeof t);
  std::memcpy(&k, &model.bytes[b.ofsKeys], sizeof k);
  t.nEntrys = 100000;
  k.nEntrys = 100000;
  model.patch(b.ofsTimes, t);
  model.patch(b.ofsKeys, k);

  Animated<float> track;
  load(track, b, model);
  CHECK(track.times[0].size() == (model.bytes.size() - t.ofsEntrys) / sizeof(uint32));
  CHECK(track.data[0].size() == 2);              // the keys end the file

  // An array offset past the end: no keys.
  k.ofsEntrys = 1u << 30;
  model.patch(b.ofsKeys, k);
  Animated<float> track2;
  load(track2, b, model);
  CHECK(!track2.uses(0));

  // Headers past the end, or more of them than fit: as many sequences as fit.
  AnimationBlock distant = b;
  distant.ofsTimes = 1u << 30;
  distant.ofsKeys = 1u << 30;
  Animated<float> track3;
  load(track3, distant, model);
  CHECK(!track3.uses(0));
  CHECK(approx(track3.getValue(0, 0), 0.0f));

  AnimationBlock many = b;
  many.nTimes = many.nKeys = 1000000;
  Animated<float> track4;
  load(track4, many, model);
  CHECK(track4.times.size() <= model.bytes.size() / sizeof(AnimationBlockHeader));

  // Times and keys disagreeing on the sequence count: the track is unusable, not misread.
  AnimationBlock odd = b;
  odd.nKeys = 2;
  Animated<float> track5;
  load(track5, odd, model);
  CHECK(!track5.uses(0));
}

void testPackedQuaternions()
{
  // Rotations are stored as four 16-bit values and converted on read: the stride through the
  // file is the stored size, not the size of the converted value.
  auto pack = [](float w, float x, float y, float z) {
    auto one = [](float v) -> int16 {
      return v > 0.0f ? int16(std::lround(v * 32767.0f) - 32768) : int16(std::lround(v * 32767.0f) + 32767);
    };
    PACK_QUATERNION q;
    q.x = one(x);
    q.y = one(y);
    q.z = one(z);
    q.w = one(w);
    return q;
  };
  const float h = std::sqrt(0.5f);
  Blob model;
  std::vector<Seq<PACK_QUATERNION> > seqs(1);
  seqs[0].times = { 0, 100 };
  seqs[0].keys = { pack(1, 0, 0, 0), pack(h, 0, 0, h) };   // identity, 90 degrees about z
  const AnimationBlock b = writeTrack(model, INTERPOLATION_LINEAR, -1, seqs);

  Animated<glm::fquat, PACK_QUATERNION, Quat16ToQuat32> rot;
  load(rot, b, model);
  CHECK(rot.data[0].size() == 2);
  const glm::fquat a = rot.getValue(0, 0);
  CHECK(approx(a.w, 1.0f, 1e-3f) && approx(a.z, 0.0f, 1e-3f));
  const glm::fquat m = rot.getValue(0, 50);       // slerp: 45 degrees
  CHECK(approx(m.w, std::cos(0.3926991f), 1e-3f) && approx(m.z, std::sin(0.3926991f), 1e-3f));
  CHECK(approx(m.x, 0.0f, 1e-3f) && approx(m.y, 0.0f, 1e-3f));
}

void testTangents()
{
  // Hermite and Bezier keys are stored value, in, out: three per key.
  Blob model;
  std::vector<Seq<float> > seqs(1);
  seqs[0].times = { 0, 100 };
  seqs[0].keys = { 1.0f, 10.0f, 11.0f, 3.0f, 30.0f, 31.0f };
  const AnimationBlock b = writeTrack(model, INTERPOLATION_BEZIER, -1, seqs);

  Animated<float> track;
  load(track, b, model);
  CHECK(track.data[0].size() == 2 && approx(track.data[0][0], 1.0f) && approx(track.data[0][1], 3.0f));
  CHECK(track.in[0].size() == 2 && approx(track.in[0][0], 10.0f) && approx(track.in[0][1], 30.0f));
  CHECK(track.out[0].size() == 2 && approx(track.out[0][0], 11.0f) && approx(track.out[0][1], 31.0f));
  CHECK(approx(track.getValue(0, 0), 1.0f));
  CHECK(approx(track.getValue(0, 100), 3.0f));
  CHECK(std::isfinite(track.getValue(0, 50)));
}

// --- aliases ---------------------------------------------------------------------------

ModelAnimation sequence(uint32 flags, int16 index)
{
  ModelAnimation a = {};
  a.flags = flags;
  a.Index = index;
  return a;
}

void testAliases()
{
  // An alias has no keys of its own and plays the sequence its Index names. Played as it
  // was, every bone fell back to its rest pose.
  std::vector<ModelAnimation> anims;
  anims.push_back(sequence(ANIMATION_KEYS_IN_MODEL, 0));   // 0: keys in the model
  anims.push_back(sequence(ANIMATION_ALIAS, 0));           // 1 -> 0
  anims.push_back(sequence(ANIMATION_ALIAS, 1));           // 2 -> 1 -> 0
  anims.push_back(sequence(0, 0));                         // 3: keys in an .anim file
  anims.push_back(sequence(ANIMATION_ALIAS, 4));           // 4: names itself
  anims.push_back(sequence(ANIMATION_ALIAS, 99));          // 5: names nothing
  anims.push_back(sequence(ANIMATION_ALIAS, 7));           // 6 -> 7 -> 6 ...
  anims.push_back(sequence(ANIMATION_ALIAS, 6));           // 7 -> 6 -> 7 ...

  CHECK(animtrack::dataSequence(anims, 0) == 0);
  CHECK(animtrack::dataSequence(anims, 1) == 0);
  CHECK(animtrack::dataSequence(anims, 2) == 0);
  CHECK(animtrack::dataSequence(anims, 3) == 3);
  CHECK(animtrack::dataSequence(anims, 4) == 4);
  CHECK(animtrack::dataSequence(anims, 5) == 5);
  const ssize_t loop = animtrack::dataSequence(anims, 6);   // returns, and stays in range
  CHECK(loop == 6 || loop == 7);
  CHECK(animtrack::dataSequence(anims, -1) == -1);
  CHECK(animtrack::dataSequence(anims, 100) == 100);
  CHECK(animtrack::dataSequence(std::vector<ModelAnimation>(), 3) == 3);
}

// --- billboards ------------------------------------------------------------------------

// The camera's axes in the model's space, the way Bone::calcMatrix takes them from the GL
// modelview: the rows of the view rotation. The camera looks at the origin from `from`, Z up.
glm::mat3 cameraAt(const glm::vec3 & from, const glm::vec3 & up = glm::vec3(0, 0, 1))
{
  return glm::transpose(glm::mat3(glm::lookAt(from, glm::vec3(0.0f), up)));
}

glm::vec3 apply(const glm::mat4 & m, const glm::vec3 & p) { return glm::vec3(m * glm::vec4(p, 1.0f)); }
glm::vec3 axis(const glm::mat4 & m, int c) { return glm::normalize(glm::vec3(m[c])); }

const glm::mat3 noSpin(1.0f);

void testBillboardFrontView()
{
  // Camera in front of the model (on +X, Z up): a billboard is drawn as built.
  const glm::mat3 front = cameraAt(glm::vec3(10, 0, 0));
  const glm::vec3 pivot(0.5f, 0.2f, 1.5f);
  const glm::mat4 full = glm::translate(glm::mat4(1.0f), glm::vec3(1, 2, 3));
  CHECK(approx(Bone::billboardMatrix(MODELBONE_BILLBOARD, full, noSpin, pivot, front), full));
  CHECK(approx(Bone::billboardMatrix(MODELBONE_BILLBOARD_LOCK_Z, full, noSpin, pivot, front), full));

  // Not a billboard: untouched, whatever the camera.
  const glm::mat3 side = cameraAt(glm::vec3(0, 10, 0));
  CHECK(approx(Bone::billboardMatrix(0, full, noSpin, pivot, side), full));
}

void testSphericalFacesCamera()
{
  // From every side, from above and below: the bone's X -- the side WoW authors a billboard
  // to show -- points at the viewer, Y runs right and Z up across the screen. The old matrix
  // flattened a glow into a line seen from the side and laid it down seen from above.
  const glm::vec3 pivot(0, 0.3f, 1.7f);
  const glm::mat4 full = glm::translate(glm::mat4(1.0f), glm::vec3(0.2f, 0, 0));
  bool facing = true, upright = true;
  for (int yaw = 0; yaw < 360; yaw += 30)
    for (int pitch = -60; pitch <= 60; pitch += 30)
    {
      const float a = glm::radians(float(yaw)), b = glm::radians(float(pitch));
      const glm::vec3 from(std::cos(b) * std::cos(a), std::cos(b) * std::sin(a), std::sin(b));
      const glm::mat3 cam = cameraAt(from * 10.0f);
      const glm::mat4 m = Bone::billboardMatrix(MODELBONE_BILLBOARD, full, noSpin, pivot, cam);
      facing = facing && approx(axis(m, 0), glm::normalize(from), 1e-3f);
      upright = upright && approx(axis(m, 1), glm::normalize(cam[0]), 1e-3f)
                        && approx(axis(m, 2), glm::normalize(cam[1]), 1e-3f);
    }
  CHECK(facing);
  CHECK(upright);

  // Straight from above, screen-up along +X: still a flat face toward the viewer.
  const glm::mat3 top = cameraAt(glm::vec3(0, 0, 10), glm::vec3(1, 0, 0));
  const glm::mat4 m = Bone::billboardMatrix(MODELBONE_BILLBOARD, full, noSpin, pivot, top);
  CHECK(approx(axis(m, 0), glm::vec3(0, 0, 1), 1e-3f));
  CHECK(approx(axis(m, 2), glm::vec3(1, 0, 0), 1e-3f));
}

void testBillboardKeepsPivotAndScale()
{
  // The pivot stays where the animation put it, and the bone keeps its size.
  const glm::vec3 pivot(0.4f, -0.2f, 1.1f);
  glm::mat4 full = glm::translate(glm::mat4(1.0f), glm::vec3(0.1f, 0.3f, -0.2f));
  full = glm::rotate(full, glm::radians(70.0f), glm::vec3(0, 0, 1));    // a parent's turn
  full = glm::scale(full, glm::vec3(2.0f, 2.0f, 2.0f));
  const glm::mat3 side = cameraAt(glm::vec3(3, 7, 2));
  for (const uint32 flags : { uint32(MODELBONE_BILLBOARD), uint32(MODELBONE_BILLBOARD_LOCK_X),
                              uint32(MODELBONE_BILLBOARD_LOCK_Y), uint32(MODELBONE_BILLBOARD_LOCK_Z) })
  {
    const glm::mat4 m = Bone::billboardMatrix(flags, full, noSpin, pivot, side);
    CHECK(approx(apply(m, pivot), apply(full, pivot)));
    CHECK(approx(glm::length(glm::vec3(m[0])), 2.0f) && approx(glm::length(glm::vec3(m[1])), 2.0f)
          && approx(glm::length(glm::vec3(m[2])), 2.0f));
    // A rotation, not a mirror: the faces keep their winding.
    CHECK(glm::determinant(glm::mat3(m)) > 0.0f);
  }
}

void testSphericalIgnoresParentsButSpins()
{
  // A parent's turn moves the billboard but does not turn it away from the viewer.
  const glm::vec3 pivot(0, 0, 1);
  const glm::mat3 front = cameraAt(glm::vec3(10, 0, 0));
  const glm::mat4 parent = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 0, 1));
  const glm::mat4 m = Bone::billboardMatrix(MODELBONE_BILLBOARD, parent, noSpin, pivot, front);
  CHECK(approx(axis(m, 0), glm::vec3(1, 0, 0), 1e-3f));

  // Its own rotation is kept: a glow spinning about its facing axis still spins on screen.
  const glm::mat3 spin = glm::mat3(glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0)));
  const glm::mat4 s = Bone::billboardMatrix(MODELBONE_BILLBOARD, glm::mat4(1.0f), spin, pivot, front);
  CHECK(approx(axis(s, 0), glm::vec3(1, 0, 0), 1e-3f));      // still facing
  CHECK(approx(axis(s, 1), glm::vec3(0, 0, 1), 1e-3f));      // its Y turned up
}

void testCylindrical()
{
  const glm::vec3 pivot(0, 0, 1);
  const glm::mat4 full(1.0f);

  // Locked to Z (an upright flame): turns about the vertical to face the viewer, but does not
  // lean back toward a camera above it. These were ignored outright -- flat cards that went
  // edge-on as the camera came round.
  bool turns = true, stays = true;
  for (int yaw = 0; yaw < 360; yaw += 30)
  {
    const float a = glm::radians(float(yaw));
    const glm::vec3 level(std::cos(a), std::sin(a), 0.0f);
    const glm::mat3 cam = cameraAt(level * 10.0f + glm::vec3(0, 0, 6));   // from above
    const glm::mat4 m = Bone::billboardMatrix(MODELBONE_BILLBOARD_LOCK_Z, full, noSpin, pivot, cam);
    turns = turns && approx(axis(m, 0), level, 1e-3f);
    stays = stays && approx(axis(m, 2), glm::vec3(0, 0, 1), 1e-3f);
  }
  CHECK(turns);
  CHECK(stays);

  // Looking straight down the locked axis no turn helps: the bone is left as it is.
  const glm::mat3 top = cameraAt(glm::vec3(0, 0, 10), glm::vec3(1, 0, 0));
  CHECK(approx(Bone::billboardMatrix(MODELBONE_BILLBOARD_LOCK_Z, full, noSpin, pivot, top), full));

  // Locked to X (a beam along X): turns about X until its Y lies flat across the screen,
  // so the strip it spans shows its face, not its edge.
  const glm::vec3 from = glm::normalize(glm::vec3(0.3f, 1.0f, 0.6f));
  const glm::mat3 cam = cameraAt(from * 10.0f);
  const glm::mat4 bx = Bone::billboardMatrix(MODELBONE_BILLBOARD_LOCK_X, full, noSpin, pivot, cam);
  CHECK(approx(axis(bx, 0), glm::vec3(1, 0, 0), 1e-3f));
  CHECK(approx(glm::dot(axis(bx, 1), from), 0.0f, 1e-3f));

  // Locked to Y, the same about Y.
  const glm::mat4 by = Bone::billboardMatrix(MODELBONE_BILLBOARD_LOCK_Y, full, noSpin, pivot, cam);
  CHECK(approx(axis(by, 1), glm::vec3(0, 1, 0), 1e-3f));
  CHECK(approx(glm::dot(axis(by, 0), from), 0.0f, 1e-3f));

  // The locked axis is the bone's as animated: a tilted bone turns about its tilted axis.
  const glm::mat4 tilted = glm::rotate(glm::mat4(1.0f), glm::radians(30.0f), glm::vec3(1, 0, 0));
  const glm::mat4 bt = Bone::billboardMatrix(MODELBONE_BILLBOARD_LOCK_Z, tilted, noSpin, pivot, cam);
  CHECK(approx(axis(bt, 2), axis(tilted, 2), 1e-3f));
}

}  // namespace

int main()
{
  testFindSegment();
  testTrackTime();
  testGlobalTrack();
  testStillGlobalHoldsFirstKey();
  testKeysPerSequence();
  testMissingAnimFile();
  testManySequences();
  testEnds();
  testCountsClampedToFile();
  testPackedQuaternions();
  testTangents();
  testAliases();
  testBillboardFrontView();
  testSphericalFacesCamera();
  testBillboardKeepsPivotAndScale();
  testSphericalIgnoresParentsButSpins();
  testCylindrical();

  if (failures == 0)
    std::printf("all animation checks passed\n");
  else
    std::printf("%d animation check(s) failed\n", failures);
  return failures;
}
