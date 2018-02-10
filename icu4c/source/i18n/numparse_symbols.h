// © 2018 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING && !UPRV_INCOMPLETE_CPP11_SUPPORT
#ifndef __NUMPARSE_SYMBOLS_H__
#define __NUMPARSE_SYMBOLS_H__

#include "numparse_types.h"
#include "unicode/uniset.h"
#include "numparse_unisets.h"

U_NAMESPACE_BEGIN namespace numparse {
namespace impl {


/**
 * A base class for many matchers that performs a simple match against a UnicodeString and/or UnicodeSet.
 *
 * @author sffc
 */
class SymbolMatcher : public NumberParseMatcher, public UMemory {
  public:
    SymbolMatcher() = default;  // WARNING: Leaves the object in an unusable state

    const UnicodeSet* getSet();

    bool match(StringSegment& segment, ParsedNumber& result, UErrorCode& status) const override;

    const UnicodeSet& getLeadCodePoints() override;

    virtual bool isDisabled(const ParsedNumber& result) const = 0;

    virtual void accept(StringSegment& segment, ParsedNumber& result) const = 0;

  protected:
    UnicodeString fString;
    const UnicodeSet* fUniSet; // a reference from numparse_unisets.h; never owned

    SymbolMatcher(const UnicodeString& symbolString, unisets::Key key);
};


class IgnorablesMatcher : public SymbolMatcher {
  public:
    IgnorablesMatcher() = default;  // WARNING: Leaves the object in an unusable state

    IgnorablesMatcher(unisets::Key key);

    bool isFlexible() const override;

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;
};


class InfinityMatcher : public SymbolMatcher {
  public:
    InfinityMatcher() = default;  // WARNING: Leaves the object in an unusable state

    InfinityMatcher(const DecimalFormatSymbols& dfs);

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;
};


class MinusSignMatcher : public SymbolMatcher {
  public:
    MinusSignMatcher() = default;  // WARNING: Leaves the object in an unusable state

    MinusSignMatcher(const DecimalFormatSymbols& dfs, bool allowTrailing);

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;

  private:
    bool fAllowTrailing;
};


class NanMatcher : public SymbolMatcher {
  public:
    NanMatcher() = default;  // WARNING: Leaves the object in an unusable state

    NanMatcher(const DecimalFormatSymbols& dfs);

    const UnicodeSet& getLeadCodePoints() override;

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;
};


class PaddingMatcher : public SymbolMatcher {
  public:
    PaddingMatcher() = default;  // WARNING: Leaves the object in an unusable state

    PaddingMatcher(const UnicodeString& padString);

    bool isFlexible() const override;

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;
};


class PercentMatcher : public SymbolMatcher {
  public:
    PercentMatcher() = default;  // WARNING: Leaves the object in an unusable state

    PercentMatcher(const DecimalFormatSymbols& dfs);

    void postProcess(ParsedNumber& result) const override;

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;
};


class PermilleMatcher : public SymbolMatcher {
  public:
    PermilleMatcher() = default;  // WARNING: Leaves the object in an unusable state

    PermilleMatcher(const DecimalFormatSymbols& dfs);

    void postProcess(ParsedNumber& result) const override;

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;
};


class PlusSignMatcher : public SymbolMatcher {
  public:
    PlusSignMatcher() = default;  // WARNING: Leaves the object in an unusable state

    PlusSignMatcher(const DecimalFormatSymbols& dfs, bool allowTrailing);

  protected:
    bool isDisabled(const ParsedNumber& result) const override;

    void accept(StringSegment& segment, ParsedNumber& result) const override;

  private:
    bool fAllowTrailing;
};


} // namespace impl
} // namespace numparse
U_NAMESPACE_END

#endif //__NUMPARSE_SYMBOLS_H__
#endif /* #if !UCONFIG_NO_FORMATTING */