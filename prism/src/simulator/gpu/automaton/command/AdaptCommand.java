package simulator.gpu.automaton.command;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import simulator.gpu.automaton.Guard;
import simulator.gpu.automaton.PrismVariable;
import simulator.gpu.automaton.update.Rate;
import simulator.gpu.automaton.update.Update;

public class AdaptCommand implements CommandInterface
{

	public final String adaptLabel;
	public final Guard guard;
	public PrismVariable var = null;
	private Map<String, Update> branches = new HashMap<>();
	private List<String> branchNames = new ArrayList<>();

	public AdaptCommand(String label, Guard guard)
	{
		adaptLabel = label;
		this.guard = guard;
	}

	public int getBranchNum()
	{
		//TODO: WORKS ONLY ON BINARY
		//TODO: fix when lower limit != 0
		if (var != null) {
			return var.maxValue + 1;
		} else {
			return 1;
		}
	}

	public void setVar(PrismVariable var)
	{
		this.var = var;
	}

	public void addCommand(String branchName, Command cmd)
	{
		branches.put(branchName, cmd.getUpdate());
		branchNames.add(branchName);
	}

	public List<String> getNames()
	{
		return branchNames;
	}

	public Update getUpdate(String name)
	{
		return branches.get(name);
	}

	public int getCommandNumber()
	{
		return branches.size();
	}

	@Override
	public Guard getGuard()
	{
		throw new IllegalAccessError("Method getGuard is not " + "defined for type SynchronizedCommand");
	}

	@Override
	public Update getUpdate()
	{
		throw new IllegalAccessError("Method getUpdate is not " + "defined for type SynchronizedCommand");
	}

	@Override
	public boolean isSynchronized()
	{
		return true;
	}

	public String toString()
	{
		StringBuilder builder = new StringBuilder();
		builder.append("ADAPT COMMAND: ").append(adaptLabel).append("\n");
		for (Map.Entry<String, Update> group : branches.entrySet()) {
			builder.append(group.getKey());
			builder.append(" update ").append(group.getValue()).append("\n");
		}
		return builder.toString();
	}

	@Override
	public Rate getRateSum()
	{
		return new Rate(branches.size());
	}

}
