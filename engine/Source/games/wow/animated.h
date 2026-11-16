#ifndef ANIMATED_H
#define ANIMATED_H

#include <algorithm>
#include <cstring>
#include <map>
#include <ostream>
#include <utility>
#include <vector>

#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"

#include "GameFile.h"
#include "modelheaders.h"
#include "types.h"


#ifdef _WIN32
#    ifdef BUILDING_WOW_DLL
#        define _ANIMATED_API_ __declspec(dllexport)
#    else
#        define _ANIMATED_API_ __declspec(dllimport)
#    endif
#else
#    define _ANIMATED_API_
#endif

// Where the keyframes of each animation sequence live. A track's per-sequence array headers
// are always in the file that defines the track -- the M2's MD21 chunk, or a skeleton's SKB1 --
// and point at their keys: in that same file, or, for a sequence stored outside the model, in
// that sequence's .anim file.
class modelAnimData
{
public:
  // Keyed by SEQUENCE INDEX, which is what a track's per-sequence arrays are indexed by. It
  // used to be keyed by animation id: every variation of an animation (Stand 0, Stand 1, ...)
  // then read its keys from whichever variation's .anim file was registered last, and a
  // sequence stored in the model read them from a sibling's .anim file. The offsets pointed
  // into the wrong buffer either way, and the bones came out as garbage.
  //
  // A sequence whose keys are in the model has no entry. One whose keys are in an .anim file
  // that could not be found has an entry of 0: it has no keys, and must not read the model
  // with offsets meant for the file.
  std::map<size_t, GameFile *> animfiles;
  std::vector<uint32> globalSequences;
};

// interpolation functions
template<class T>
inline T interpolate(const float r, const T &v1, const T &v2)
{
  return static_cast<T>(v1*(1.0f - r) + v2*r);
}

template<class T>
inline T interpolateHermite(const float r, const T &v1, const T &v2, const T &in, const T &out)
{
  // basis functions
  float h1 = 2.0f*r*r*r - 3.0f*r*r + 1.0f;
  float h2 = -2.0f*r*r*r + 3.0f*r*r;
  float h3 = r*r*r - 2.0f*r*r + r;
  float h4 = r*r*r - r*r;

  // interpolation
  return static_cast<T>(v1*h1 + v2*h2 + in*h3 + out*h4);
}


template<class T>
inline T interpolateBezier(const float r, const T &v1, const T &v2, const T &in, const T &out)
{
  float InverseFactor = (1.0f - r);
  float FactorTimesTwo = r*r;
  float InverseFactorTimesTwo = InverseFactor*InverseFactor;
  // basis functions
  float h1 = InverseFactorTimesTwo * InverseFactor;
  float h2 = 3.0f * r * InverseFactorTimesTwo;
  float h3 = 3.0f * FactorTimesTwo * InverseFactor;
  float h4 = FactorTimesTwo * r;

  // interpolation
  return static_cast<T>(v1*h1 + v2*h2 + in*h3 + out*h4);
}

// "linear" interpolation for quaternions should be slerp by default
template<>
inline glm::fquat interpolate<glm::fquat>(const float r, const  glm::fquat &v1, const  glm::fquat &v2)
{
  return glm::slerp(v1, v2, r);
}


typedef std::pair<size_t, size_t> AnimRange;

// global time for global sequences
_ANIMATED_API_ extern size_t globalTime;

enum Interpolations {
  INTERPOLATION_NONE,
  INTERPOLATION_LINEAR,
  INTERPOLATION_HERMITE,
  INTERPOLATION_BEZIER
};

// The time arithmetic of a track, apart from the class so the tests can reach it.
namespace animtrack
{
// The clock a track runs on. A track bound to a global sequence runs on the model-wide clock,
// wrapped to the sequence's length. A global sequence of length 0 holds its first key, as the
// game does; so does an index past the end of the list. Everything else runs on the time
// within the playing sequence.
//
// A length of 0 used to return a default-constructed value instead of the first key: a zero
// vector -- which for a SCALE track collapses every vertex of the bone onto its pivot -- and an
// index past the end fell through to the per-sequence arrays, which a global track does not
// have, with the same result.
inline size_t trackTime(ssize_t seq, const std::vector<uint32> & globals, size_t sequenceTime,
                        size_t globalClock)
{
  if (seq < 0)
    return sequenceTime;
  const uint32 length = (seq < (ssize_t)globals.size()) ? globals[seq] : 0;
  return length ? globalClock % length : 0;
}

// The keyframe segment `time` falls in: its first key, and how far into it (0..1). Before the
// first key and from the last one on, the end key holds.
//
// The search used to start every lookup at segment 0 and keep it when nothing matched, so a
// time before the first key -- or exactly on the last one -- extrapolated the first segment by
// an unbounded factor. Before the first key it was worse: time - t1 underflowed in size_t and
// the ratio came out near 1.8e19, which flings a bone and everything skinned to it to
// infinity. Two keys on the same time divided by zero.
struct Segment
{
  size_t index;
  float ratio;
};

inline Segment findSegment(const std::vector<size_t> & times, size_t count, size_t time)
{
  if (count < 2 || time <= times[0])
    return { 0, 0.0f };
  if (time >= times[count - 1])
    return { count - 1, 0.0f };

  size_t i = size_t(std::upper_bound(times.begin(), times.begin() + count, time) - times.begin());
  i = (i == 0) ? 0 : i - 1;
  if (i > count - 2)          // unsorted keys from a malformed file
    i = count - 2;
  const size_t t1 = times[i];
  const size_t t2 = times[i + 1];
  if (t2 <= t1 || time < t1)
    return { i, 0.0f };
  return { i, float(time - t1) / float(t2 - t1) };
}

// The sequence whose keys play when sequence `index` plays. An alias (ANIMATION_ALIAS) has no
// keys of its own and stands for the sequence its Index names, which may be an alias in turn.
// Every other sequence -- and anything out of range -- is its own. The walk is bounded, so an
// alias loop in a malformed file cannot hang the frame.
//
// Aliases used to be played as they were: every bone track of them is empty, so each bone
// fell back to its rest pose -- or, for a scale track, to a zero vector that collapsed it.
inline ssize_t dataSequence(const std::vector<ModelAnimation> & anims, ssize_t index)
{
  for (size_t hops = 0; hops < anims.size(); hops++)
  {
    if (index < 0 || index >= (ssize_t)anims.size() || !(anims[index].flags & ANIMATION_ALIAS))
      return index;
    const ssize_t next = anims[index].Index;
    if (next < 0 || next >= (ssize_t)anims.size() || next == index)
      return index;
    index = next;
  }
  return index;
}
}

template <class T>
class Identity {
public:
  static const T& conv(const T& t)
  {
    return t;
  }
};

// In WoW 2.0+ Blizzard are now storing rotation data in 16bit values instead of 32bit.
// I don't really understand why as its only a very minor saving in model sizes and adds extra overhead in
// processing the models.  Need this structure to read the data into.
struct PACK_QUATERNION {
  int16 x,y,z,w;
};

class Quat16ToQuat32 {
public:
  static const glm::fquat conv(const PACK_QUATERNION t)
  {
    return glm::fquat(
      float(t.w < 0 ? t.w + 32768 : t.w - 32767) / 32767.0f,
      float(t.x < 0? t.x + 32768 : t.x - 32767)/ 32767.0f,
      float(t.y < 0? t.y + 32768 : t.y - 32767)/ 32767.0f,
      float(t.z < 0? t.z + 32768 : t.z - 32767)/ 32767.0f);
  }
};

// Convert opacity values stored as shorts to floating point
// I wonder why Blizzard decided to save 2 bytes by doing this
class ShortToFloat {
public:
  static float conv(const short t)
  {
    return t/32767.0f;
  }
};

/*
  Generic animated value class:

  T is the data type to animate
  D is the data type stored in the file (by default this is the same as T)
  Conv is a conversion object that defines T conv(D) to convert from D to T
    (by default this is an identity function)
  (there might be a nicer way to do this? meh meh)
*/

template <class T, class D=T, class Conv=Identity<T> >
class Animated {
public:

  ssize_t type = INTERPOLATION_NONE, seq = -1;
  std::vector<uint32> globals;
  // One array per animation sequence, indexed like the model's sequences -- or a single one
  // for a track on a global sequence. Sized to what the track declares: they used to be fixed
  // arrays of 500, which silently dropped every sequence past the 500th (modern characters
  // have more) and cost 48 KB per track even when it held a single key.
  std::vector<std::vector<size_t> > times;
  std::vector<std::vector<T> > data;
  // for nonlinear interpolations:
  std::vector<std::vector<T> > in, out;
  size_t sizes = 0; // for fix function

  bool uses(ssize_t anim) const
  {
    if (seq>-1)
      anim = 0;
    return anim >= 0 && anim < (ssize_t)data.size() && !data[anim].empty();
  }

  T getValue(ssize_t anim, size_t time) const
  {
    if (seq > -1)
    {
      time = animtrack::trackTime(seq, globals, time, globalTime);
      anim = 0;
    }
    if (anim < 0 || anim >= (ssize_t)data.size() || data[anim].empty())
      return T();

    const std::vector<T> & keys = data[anim];
    const size_t count = std::min(keys.size(), times[anim].size());
    if (count < 2)
      return keys[0];

    const animtrack::Segment s = animtrack::findSegment(times[anim], count, time);
    const size_t i = s.index;
    if (i + 1 >= count || type == INTERPOLATION_NONE)
      return keys[i];

    switch (type)
    {
      case INTERPOLATION_LINEAR:
        return interpolate<T>(s.ratio, keys[i], keys[i + 1]);
      case INTERPOLATION_HERMITE:
      case INTERPOLATION_BEZIER:
        // tangents read short of the keys (a truncated file): fall back to a straight line
        if (anim >= (ssize_t)in.size() || i >= in[anim].size() || i >= out[anim].size())
          return interpolate<T>(s.ratio, keys[i], keys[i + 1]);
        if (type == INTERPOLATION_HERMITE) // only used in cameras afaik?
          return interpolateHermite<T>(s.ratio, keys[i], keys[i + 1], in[anim][i], out[anim][i]);
        return interpolateBezier<T>(s.ratio, keys[i], keys[i + 1], in[anim][i], out[anim][i]);
      default: //this shouldn't appear!
        return keys[i];
    }
  }

  // Tracks that never have .anim files: particles, ribbons, colours, transparency, texture
  // animations, lights, cameras.
  void init(AnimationBlock &b, GameFile * f, std::vector<uint32> & gs)
  {
    load(b, gs, f->getBuffer(), f->getSize(),
         [](size_t, const unsigned char *&, size_t &) { return true; });
  }

  // Bone tracks. `f` is the file that defines the bones, positioned at MD21 or SKB1; a
  // sequence stored outside the model has its keys in its .anim file instead.
  void init(AnimationBlock &b, GameFile & f, const modelAnimData & modelData)
  {
    load(b, modelData.globalSequences, f.getBuffer(), f.getSize(),
         [&modelData](size_t j, const unsigned char *& buf, size_t & bufSize)
         {
           auto it = modelData.animfiles.find(j);
           if (it == modelData.animfiles.end())
             return true;      // keys in the model
           if (!it->second)
             return false;     // keys in an .anim file that is not there: none
           buf = it->second->getBuffer();
           bufSize = it->second->getSize();
           return true;
         });
  }

  // What both init()s do, on plain bytes so the tests can feed it without game files. `file`
  // holds the track's per-sequence array headers. keysFor(j, buf, size) may point sequence j's
  // keys at another buffer -- its .anim file -- or return false when the sequence has none.
  template <class KeysFor>
  void load(const AnimationBlock & b, const std::vector<uint32> & gs,
            const unsigned char * file, size_t fileSize, KeysFor keysFor)
  {
    globals = gs;
    begin(b, fileSize);
    for (size_t j = 0; j < times.size(); j++)
    {
      const unsigned char * buf = file;
      size_t bufSize = fileSize;
      // A track on a global sequence is not per-sequence data and is never in an .anim file;
      // reading it from sequence 0's file read garbage whenever that sequence had one.
      if (seq < 0 && !keysFor(j, buf, bufSize))
        continue;
      readSequence(j, b, file, fileSize, buf, bufSize);
    }
  }

  friend std::ostream& operator<<(std::ostream& out, const Animated& v)
  {
    if (v.sizes == 0)
      return out;
    out << "      <type>"<< v.type << "</type>" << std::endl;
    out << "      <seq>"<< v.seq << "</seq>" << std::endl;
    out << "      <anims>"<< std::endl;
    for(size_t j=0; j<v.data.size(); j++) {
      if (j != 0) continue; // only output walk animation
      if (v.uses((unsigned int)j)) {
        out << "    <anim id=\"" << j << "\" size=\""<< v.data[j].size() <<"\">" << std::endl;
        out << "    </anim>" << std::endl;
      }
      if (v.seq > -1 && j > 0)
        break;
    }
    out << "      </anims>"<< std::endl;
    return out;
  }

private:
  // One array per declared sequence, as many as the track's headers fit in the file.
  void begin(const AnimationBlock & b, size_t headerFileSize)
  {
    type = b.type;
    seq = b.seq;
    times.clear();
    data.clear();
    in.clear();
    out.clear();
    sizes = 0;

    if (b.nTimes != b.nKeys || b.nTimes == 0)
      return;

    const size_t hsz = sizeof(AnimationBlockHeader);
    size_t n = b.nTimes;
    const size_t firstHeader = std::max<size_t>(b.ofsTimes, b.ofsKeys);
    if (firstHeader >= headerFileSize)
      return;
    n = std::min(n, (headerFileSize - firstHeader) / hsz);

    sizes = n;
    times.resize(n);
    data.resize(n);
    if (type == INTERPOLATION_HERMITE || type == INTERPOLATION_BEZIER)
    {
      in.resize(n);
      out.resize(n);
    }
  }

  // Sequence j: its array headers in `hdr` (the file that defines the track), the arrays in
  // `buf` -- the same file, or the sequence's .anim file. Counts are clamped to what the
  // buffer holds: a header that over-reports (a modern rig, an offset near the end) used to
  // read past the buffer and turn the bone's track into garbage.
  void readSequence(size_t j, const AnimationBlock & b, const unsigned char * hdr, size_t hdrSize,
                    const unsigned char * buf, size_t bufSize)
  {
    const size_t hsz = sizeof(AnimationBlockHeader);
    if ((size_t)b.ofsTimes + (j + 1) * hsz > hdrSize || (size_t)b.ofsKeys + (j + 1) * hsz > hdrSize)
      return;

    AnimationBlockHeader headTimes, headKeys;
    memcpy(&headTimes, hdr + b.ofsTimes + j * hsz, hsz);
    memcpy(&headKeys, hdr + b.ofsKeys + j * hsz, hsz);

    if (headTimes.ofsEntrys <= bufSize)
    {
      const size_t n = std::min<size_t>(headTimes.nEntrys,
                                        (bufSize - headTimes.ofsEntrys) / sizeof(uint32));
      const uint32 * src = (const uint32 *)(buf + headTimes.ofsEntrys);
      times[j].reserve(n);
      for (size_t i = 0; i < n; i++)
        times[j].push_back(src[i]);
    }

    if (headKeys.ofsEntrys <= bufSize)
    {
      // Hermite and Bezier store three values per key: the value and its two tangents.
      const bool tangents = (type == INTERPOLATION_HERMITE || type == INTERPOLATION_BEZIER);
      const size_t perKey = tangents ? 3 : 1;
      const size_t n = std::min<size_t>(headKeys.nEntrys,
                                        (bufSize - headKeys.ofsEntrys) / (sizeof(D) * perKey));
      const D * keys = (const D *)(buf + headKeys.ofsEntrys);
      data[j].reserve(n);
      for (size_t i = 0; i < n; i++)
      {
        data[j].push_back(Conv::conv(keys[i * perKey]));
        if (tangents)
        {
          in[j].push_back(Conv::conv(keys[i * 3 + 1]));
          out[j].push_back(Conv::conv(keys[i * 3 + 2]));
        }
      }
    }
  }
};

typedef Animated<float,short,ShortToFloat> AnimatedShort;

float frand();

float randfloat(float lower, float upper);
_ANIMATED_API_ int randint(int lower, int upper);

#endif
