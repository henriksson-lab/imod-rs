$batchruntomo -StandardInput
NamingStyle 1
MakeSubDirectory
CoresPerClusterJob 4
GPUsPerClusterJob 1
DirectiveFile  d1.adoc
RootName ts1
CurrentLocation /data/a
DeliverToDirectory /data/out1
DirectiveFile  d2.adoc
RootName ts2
CurrentLocation /data/b
DeliverToDirectory /data/out2
CheckFile batch.cmds
EmailAddress x@y
