#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mxml.h>

static FILE *OUT;
static int sLastLevel = -1;
static char sWs[36];
static mxml_node_t *nodes[100000];
static int numNodes;

static const char *ws_cb(mxml_node_t *node, int where)
{
  int level = -1;
  mxml_node_t *parent = node->parent;
  char spaces[33];
  memset(spaces, ' ', 33);
  if (where != MXML_WS_BEFORE_OPEN && where != MXML_WS_BEFORE_CLOSE) return NULL;
  while (parent) { level++; parent = parent->parent; }
  if (level > 16) level = 16; else if (level < 0) level = 0;
  if (sLastLevel < 0) { sLastLevel = level; return NULL; }
  if (level == sLastLevel && where == MXML_WS_BEFORE_CLOSE) return NULL;
  sLastLevel = level;
  spaces[32] = 0x00;
  snprintf(sWs, 36, "\n%s", &spaces[32 - 2 * level]);
  return sWs;
}

static int node_index(mxml_node_t *n)
{
  int i;
  if (!n) return -1;
  for (i = 0; i < numNodes; i++) if (nodes[i] == n) return i;
  return -2;
}

static void dump_tree(mxml_node_t *tree, const char *tag)
{
  int i, k, ws = 0;
  mxml_node_t *n;
  numNodes = 0;
  for (n = tree; n; n = mxmlWalkNext(n, tree, MXML_DESCEND)) nodes[numNodes++] = n;
  fprintf(OUT, "%s nodes=%d\n", tag, numNodes);
  for (i = 0; i < numNodes; i++) {
    n = nodes[i];
    fprintf(OUT, "  [%d] type=%d parent=%d child=%d last=%d prev=%d next=%d ref=%d",
            i, mxmlGetType(n), node_index(n->parent), node_index(n->child),
            node_index(n->last_child), node_index(n->prev), node_index(n->next),
            mxmlGetRefCount(n));
    switch (n->type) {
    case MXML_ELEMENT:
      fprintf(OUT, " elem=<%s> nattr=%d", n->value.element.name, n->value.element.num_attrs);
      for (k = 0; k < n->value.element.num_attrs; k++)
        fprintf(OUT, " attr[%d]=%s=\"%s\"", k, n->value.element.attrs[k].name,
                n->value.element.attrs[k].value ? n->value.element.attrs[k].value : "(nil)");
      if (mxmlGetCDATA(n)) fprintf(OUT, " cdata=<%s>", mxmlGetCDATA(n));
      break;
    case MXML_OPAQUE:  fprintf(OUT, " opaque=<%s>", n->value.opaque); break;
    case MXML_TEXT:    fprintf(OUT, " text=<%s> ws=%d", n->value.text.string, n->value.text.whitespace); break;
    case MXML_INTEGER: fprintf(OUT, " integer=%d", n->value.integer); break;
    case MXML_REAL:    fprintf(OUT, " real=%f", n->value.real); break;
    default: break;
    }
    fprintf(OUT, " getElem=%s getOpaque=%s getInt=%d getReal=%f getText=%s",
            mxmlGetElement(n) ? mxmlGetElement(n) : "(nil)",
            mxmlGetOpaque(n) ? mxmlGetOpaque(n) : "(nil)",
            mxmlGetInteger(n), mxmlGetReal(n),
            mxmlGetText(n, &ws) ? mxmlGetText(n, &ws) : "(nil)");
    fprintf(OUT, "\n");
  }
}

static void save_report(mxml_node_t *tree, const char *tag, mxml_save_cb_t cb, int wrap)
{
  static char buf[400000];
  char small[40];
  char *alloc;
  int n, m;
  mxmlSetWrapMargin(wrap);
  sLastLevel = -1;
  n = mxmlSaveString(tree, buf, 400000, cb);
  fprintf(OUT, "%s saveString=%d\n[%s]\n", tag, n, buf);
  sLastLevel = -1;
  alloc = mxmlSaveAllocString(tree, cb);
  fprintf(OUT, "%s saveAlloc=%s\n", tag, alloc ? "ok" : "(nil)");
  if (alloc) { fprintf(OUT, "[%s]\n", alloc); free(alloc); }
  sLastLevel = -1;
  memset(small, 0, 40);
  m = mxmlSaveString(tree, small, 40, cb);
  fprintf(OUT, "%s saveSmall=%d [%s]\n", tag, m, small);
  mxmlSetWrapMargin(72);
}

static const char *CASES[] = {
  "<a>b</a>",
  "<?xml version=\"1.0\"?><root><x>1</x></root>",
  "<!DOCTYPE foo><root/>",
  "<root><!-- a comment --><x/></root>",
  "<root><![CDATA[some <raw> data]]></root>",
  "<root a='1' b=\"two\" c=unquoted>text</root>",
  "<root>&amp;&lt;&gt;&quot;&apos;&#65;&#x42;&nbsp;</root>",
  "<root>&bogus;</root>",
  "<root><a></b></root>",
  "<root",
  "",
  "<root>unclosed",
  "<a/><b/>",
  "<root>   spaced   text   </root>",
  "<root><child x=\"1\"/><child x=\"2\"/><child x=\"3\"/></root>",
};
#define NCASES ((int)(sizeof(CASES)/sizeof(CASES[0])))

static void probe_strings(void)
{
  static char buf[8192];
  int i, t;
  const char *tags[4] = { "=== STRING-TEXTCB %d\n", "=== STRING-INTCB %d\n",
                          "=== STRING-REALCB %d\n", "=== STRING-IGNORECB %d\n" };
  mxml_load_cb_t cbs[4] = { NULL, MXML_INTEGER_CALLBACK, MXML_REAL_CALLBACK, MXML_IGNORE_CALLBACK };
  for (i = 0; i < NCASES; i++) {
    mxml_node_t *tree;
    fprintf(OUT, "=== STRING %d [%s]\n", i, CASES[i]);
    fflush(OUT);
    tree = mxmlLoadString(NULL, CASES[i], MXML_OPAQUE_CALLBACK);
    if (!tree) { fprintf(OUT, "  load NULL\n"); continue; }
    dump_tree(tree, "  TREE");
    sLastLevel = -1;
    memset(buf, 0, 8192);
    fprintf(OUT, "  save=%d [%s]\n", mxmlSaveString(tree, buf, 8192, NULL), buf);
    mxmlDelete(tree);
  }
  for (t = 0; t < 4; t++) {
    for (i = 0; i < NCASES; i++) {
      mxml_node_t *tree;
      fprintf(OUT, tags[t], i);
      fflush(OUT);
      tree = mxmlLoadString(NULL, CASES[i], cbs[t]);
      if (!tree) { fprintf(OUT, "  load NULL\n"); continue; }
      dump_tree(tree, "  TREE");
      mxmlDelete(tree);
    }
  }
}

static void probe_build(void)
{
  static char buf[8192];
  int i = 0;
  mxml_node_t *xml, *top, *elem, *t;
  int r1, r2, r3, r4, r5, a, sc;
  const char *c0;

  fprintf(OUT, "=== BUILD\n");
  xml = mxmlNewXML("1.0");
  dump_tree(xml, "  XMLONLY");
  top = mxmlNewElement(xml, "root");
  elem = mxmlNewElement(top, "child");
  mxmlElementSetAttr(elem, "one", "1");
  mxmlElementSetAttr(elem, "two", "2");
  mxmlElementSetAttr(elem, "one", "1b");
  mxmlElementSetAttr(elem, "nil", NULL);
  mxmlElementSetAttrf(elem, "fmt", "%d", 42);
  mxmlNewText(elem, 0, "hello & <world>");
  mxmlNewText(top, 1, "spaced");
  mxmlNewInteger(top, 17);
  mxmlNewReal(top, 3.5);
  mxmlNewOpaque(top, "opaque \"text\"");
  mxmlNewCDATA(top, "cdata & stuff");
  dump_tree(xml, "  BUILT");
  fprintf(OUT, "  getAttr(one)=%s\n", mxmlElementGetAttr(elem, "one"));
  fprintf(OUT, "  getAttr(nil)=%s\n", mxmlElementGetAttr(elem, "nil") ? "nonnull" : "(nil)");
  fprintf(OUT, "  getAttr(zzz)=%s\n", mxmlElementGetAttr(elem, "zzz") ? "nonnull" : "(nil)");
  mxmlElementDeleteAttr(elem, "two");
  mxmlElementDeleteAttr(elem, "zzz");
  dump_tree(xml, "  AFTERDEL");
  save_report(xml, "  BUILDSAVE", NULL, 72);
  save_report(xml, "  BUILDSAVEWS", ws_cb, 0);

  t = mxmlNewElement(NULL, "orphan");
  r1 = mxmlRetain(t); r2 = mxmlRetain(t);
  r3 = mxmlRelease(t); r4 = mxmlRelease(t); r5 = mxmlRelease(t);
  fprintf(OUT, "  retain=%d retain=%d release=%d release=%d releaseFinal=%d\n", r1, r2, r3, r4, r5);
  fprintf(OUT, "  retainNull=%d releaseNull=%d\n", mxmlRetain(NULL), mxmlRelease(NULL));

  t = mxmlNewElement(NULL, "setme");
  fprintf(OUT, "  setElement=%d\n", mxmlSetElement(t, "renamed"));
  fprintf(OUT, "  name=%s\n", mxmlGetElement(t));
  fprintf(OUT, "  setCDATAbad=%d\n", mxmlSetCDATA(t, "x"));
  fprintf(OUT, "  setUserData=%d\n", mxmlSetUserData(t, (void *)1));
  fprintf(OUT, "  getUserData=%s\n", mxmlGetUserData(t) ? "(set)" : "(nil)");
  mxmlDelete(t);

  t = mxmlNewInteger(NULL, 5);
  a = mxmlSetInteger(t, 9);
  fprintf(OUT, "  setInteger=%d %d\n", a, mxmlGetInteger(t));
  fprintf(OUT, "  setReal=%d\n", mxmlSetReal(t, 1.0));
  mxmlDelete(t);
  t = mxmlNewReal(NULL, 5.0);
  a = mxmlSetReal(t, 9.5);
  fprintf(OUT, "  setReal=%d %f\n", a, mxmlGetReal(t));
  mxmlDelete(t);
  t = mxmlNewText(NULL, 0, "a");
  fprintf(OUT, "  setText=%d\n", mxmlSetText(t, 1, "bcd"));
  fprintf(OUT, "  getText=%s\n", mxmlGetText(t, &i));
  mxmlDelete(t);
  t = mxmlNewOpaque(NULL, "a");
  a = mxmlSetOpaque(t, "xyz");
  fprintf(OUT, "  setOpaque=%d %s\n", a, mxmlGetOpaque(t));
  mxmlDelete(t);
  t = mxmlNewCDATA(NULL, "abc");
  c0 = strdup(mxmlGetCDATA(t));
  sc = mxmlSetCDATA(t, "def");
  fprintf(OUT, "  cdata=%s setCDATA=%d then=%s\n", c0, sc, mxmlGetCDATA(t));
  sLastLevel = -1;
  memset(buf, 0, 8192);
  fprintf(OUT, "  cdataSave=%d [%s]\n", mxmlSaveString(t, buf, 8192, NULL), buf);
  mxmlDelete(t);

  mxmlDelete(xml);
}

static void probe_entities(void)
{
  const char *names[] = { "amp","lt","gt","quot","apos","nbsp","AElig","zwnj","Alpha","euro",
                          "bogus","","a","zzzz" };
  int i;
  fprintf(OUT, "=== ENTITIES\n");
  for (i = 0; i < 300; i++) {
    const char *n = mxmlEntityGetName(i);
    if (n) fprintf(OUT, "  getName(%d)=%s\n", i, n);
  }
  for (i = 0; i < 14; i++)
    fprintf(OUT, "  getValue(%s)=%d\n", names[i], mxmlEntityGetValue(names[i]));
}

static void probe_file(const char *path, const char *base, const char *savedir)
{
  FILE *fp;
  mxml_node_t *tree, *n;
  mxml_index_t *ind;
  int count;
  char savepath[1024];
  fprintf(OUT, "=== FILE %s\n", base);
  fp = fopen(path, "r");
  if (!fp) { fprintf(OUT, "  no open\n"); return; }
  tree = mxmlLoadFile(NULL, fp, MXML_OPAQUE_CALLBACK);
  fclose(fp);
  if (!tree) { fprintf(OUT, "  load NULL\n"); return; }
  dump_tree(tree, "  TREE");
  fprintf(OUT, "  walkNext(tree,tree,DESCEND) idx=%d\n", node_index(mxmlWalkNext(tree, tree, MXML_DESCEND)));
  fprintf(OUT, "  walkNext(tree,tree,NO_DESCEND) idx=%d\n", node_index(mxmlWalkNext(tree, tree, MXML_NO_DESCEND)));
  n = tree;
  while (mxmlWalkNext(n, tree, MXML_DESCEND)) n = mxmlWalkNext(n, tree, MXML_DESCEND);
  fprintf(OUT, "  lastNode=%d walkPrev=%d walkPrevNoDesc=%d\n", node_index(n),
          node_index(mxmlWalkPrev(n, tree, MXML_DESCEND)),
          node_index(mxmlWalkPrev(n, tree, MXML_NO_DESCEND)));
  count = 0;
  n = mxmlFindElement(tree, tree, "Field", NULL, NULL, MXML_DESCEND);
  while (n) {
    if (count < 5) {
      const char *a = mxmlElementGetAttr(n, "name");
      fprintf(OUT, "  findField[%d]=%d attr=%s\n", count, node_index(n), a ? a : "(nil)");
    }
    count++;
    n = mxmlFindElement(n, tree, "Field", NULL, NULL, MXML_DESCEND);
  }
  fprintf(OUT, "  findFieldCount=%d\n", count);
  fprintf(OUT, "  findPath(autodoc/PreData/Version)=%d\n", node_index(mxmlFindPath(tree, "autodoc/PreData/Version")));
  fprintf(OUT, "  findPath(*/short)=%d\n", node_index(mxmlFindPath(tree, "*/short")));
  fprintf(OUT, "  findPath(nosuch)=%d\n", node_index(mxmlFindPath(tree, "nosuch")));
  ind = mxmlIndexNew(tree, "Field", "name");
  if (!ind) fprintf(OUT, "  index NULL\n");
  else {
    fprintf(OUT, "  indexCount=%d alloc=%d\n", mxmlIndexGetCount(ind), ind->alloc_nodes);
    count = 0;
    n = mxmlIndexReset(ind);
    while (n) {
      if (count < 8) {
        const char *a = mxmlElementGetAttr(n, "name");
        fprintf(OUT, "  indexEnum[%d]=%d %s\n", count, node_index(n), a ? a : "(nil)");
      }
      count++;
      n = mxmlIndexEnum(ind);
    }
    fprintf(OUT, "  indexEnumCount=%d\n", count);
    mxmlIndexReset(ind);
    fprintf(OUT, "  indexFind(Field,InputFile)=%d\n", node_index(mxmlIndexFind(ind, "Field", "InputFile")));
    mxmlIndexReset(ind);
    fprintf(OUT, "  indexFind(Field,zzz-none)=%d\n", node_index(mxmlIndexFind(ind, "Field", "zzz-none")));
    mxmlIndexDelete(ind);
  }
  ind = mxmlIndexNew(tree, NULL, NULL);
  fprintf(OUT, "  indexAllCount=%d\n", ind ? mxmlIndexGetCount(ind) : -1);
  if (ind) mxmlIndexDelete(ind);
  save_report(tree, "  SAVE-nocb", NULL, 72);
  save_report(tree, "  SAVE-ws", ws_cb, 0);
  save_report(tree, "  SAVE-wrap20", NULL, 20);
  snprintf(savepath, 1024, "%s/%s.save", savedir, base);
  fp = fopen(savepath, "w");
  if (fp) {
    mxmlSetWrapMargin(0);
    sLastLevel = -1;
    fprintf(OUT, "  saveFile=%d\n", mxmlSaveFile(tree, fp, ws_cb));
    fclose(fp);
    mxmlSetWrapMargin(72);
  }
  mxmlDelete(tree);
}

int main(int argc, char **argv)
{
  char line[4096];
  FILE *lf;
  OUT = fopen(argv[1], "w");
  probe_entities();
  probe_strings();
  probe_build();
  lf = fopen(argv[3], "r");
  while (lf && fgets(line, 4096, lf)) {
    char *nl = strchr(line, '\n');
    char *base;
    if (nl) *nl = 0;
    if (!*line) continue;
    base = strrchr(line, '/');
    base = base ? base + 1 : line;
    probe_file(line, base, argv[2]);
  }
  fclose(OUT);
  return 0;
}
