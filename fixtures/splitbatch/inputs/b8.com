$batchruntomo -StandardInput
NamingStyle 1
MakeSubDirectory
CPUMachineList localhost:4,server2:2
CheckFile batch.cmds
EmailAddress x@y
