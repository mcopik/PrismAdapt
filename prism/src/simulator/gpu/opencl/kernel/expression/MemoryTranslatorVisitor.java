//==============================================================================
//	
//	Copyright (c) 2002-
//	Authors:
//	* Marcin Copik <mcopik@gmail.com> (Silesian University of Technology)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================
package simulator.gpu.opencl.kernel.expression;

import java.util.ArrayList;
import java.util.List;

import prism.Pair;
import prism.Preconditions;
import simulator.gpu.opencl.kernel.memory.CLVariable;
import simulator.gpu.opencl.kernel.memory.StructureType;

public class MemoryTranslatorVisitor implements VisitorInterface
{
	private StructureType stateVector = null;
	private CLVariable svInstance = null;
	private List<Pair<String, String>> translations = new ArrayList<>();

	public MemoryTranslatorVisitor(StructureType sv)
	{
		stateVector = sv;
	}

	public void addTranslation(String variableName, String fieldName)
	{
		translations.add(new Pair<>(variableName, fieldName));
	}

	private String translateString(String str)
	{
		Preconditions.checkNotNull(svInstance, "Visiting without specyfing SV instance!");
		for (Pair<String, String> pair : translations) {
			//enable easy debugging!
			if (!str.contains("printf")) {
				CLVariable newVar = svInstance.varType.accessField(svInstance.varName, pair.second);
				str = str.replaceAll("\\b" + pair.first + "\\b", newVar.varName);
			}
		}
		return str;
	}

	public void setStateVector(CLVariable var)
	{
		Preconditions.checkCondition(var.varType.isStructure(), "Setting access to SV with non-structure variable!");
		svInstance = var;
	}

	@Override
	public void visit(Expression expr)
	{
		expr.exprString = translateString(expr.exprString);
	}

	@Override
	public void visit(Method method)
	{

	}

	@Override
	public void visit(ForLoop loop)
	{

	}
}